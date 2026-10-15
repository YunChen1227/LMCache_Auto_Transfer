# SPDX-License-Identifier: Apache-2.0
# Standard
import threading

# Third Party
import pytest

# First Party
from nas_kv_warmup.config import WarmupConfig
from nas_kv_warmup.journal import WarmSequence
from nas_kv_warmup.replayer import (
    ReplayConfig,
    WarmupReplayer,
    build_completion_body,
)


def seq(name, n_chunks=2, ts=1.0, model="m", ws=2, salt=""):
    return WarmSequence(
        model=model,
        ws=ws,
        salt=salt,
        token_ids=tuple(range(n_chunks * 4)),
        chunk_hashes=tuple(f"{name}{i}" for i in range(n_chunks)),
        ts=ts,
    )


class FakeBackend:
    def __init__(self, registered=(("m", 2),), missing=(), pending_polls=1):
        self.registered = set(registered)
        self.missing = set(missing)
        self.pending_polls = pending_polls
        self.polls = {}
        self.submitted = []
        self.lock = threading.Lock()

    def is_registered(self, model, world_size):
        return (model, world_size) in self.registered

    def submit_prefetch(self, s):
        with self.lock:
            self.submitted.append(s.chunk_hashes[0][:-1])
        if s.chunk_hashes[0].startswith("noop"):
            return None
        job_id = s.chunk_hashes[0]
        self.polls[job_id] = 0
        return job_id

    def poll_prefetch(self, job_id):
        self.polls[job_id] += 1
        if self.polls[job_id] <= self.pending_polls:
            return None
        if job_id[:-1] in self.missing:
            return (1, 4)
        return (4, 4)


class FakeClient:
    def __init__(self, healthy_after=0, fail=()):
        self.health_calls = 0
        self.healthy_after = healthy_after
        self.fail = set(fail)
        self.bodies = []
        self.lock = threading.Lock()

    def health(self):
        self.health_calls += 1
        return self.health_calls > self.healthy_after

    def complete(self, body):
        with self.lock:
            self.bodies.append(body)
        if body["prompt"][-1] in self.fail:
            raise RuntimeError("boom")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.5
        return self.t


def config(**kw):
    defaults = dict(poll_interval_s=0.001, prefetch_batch_size=2)
    defaults.update(kw)
    return ReplayConfig(**defaults)


def test_completion_body():
    body = build_completion_body(seq("a", ts=1, salt="t"), "served")
    assert body == {
        "model": "served",
        "prompt": list(range(8)),
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "cache_salt": "t",
    }
    assert build_completion_body(seq("a"), None)["model"] == "m"
    assert "cache_salt" not in build_completion_body(seq("a"), None)


def test_replays_found_sequences_oldest_first_and_skips_missing():
    backend = FakeBackend(missing={"b"})
    client = FakeClient(healthy_after=2)
    seqs = [seq("a", ts=3), seq("b", ts=2), seq("c", ts=1), seq("noop", ts=0.5)]
    replayer = WarmupReplayer(seqs, backend, client, config(replay_concurrency=1))
    replayer.run()

    stats = replayer.stats()
    assert stats["state"] == "done"
    assert stats["replayed"] == 2
    assert stats["skipped_missing"] == 2
    assert stats["tokens_warmed"] == 16
    assert backend.submitted == ["noop", "c", "b", "a"]
    assert all(b["max_tokens"] == 1 for b in client.bodies)
    assert all(b["prompt"] == list(range(8)) for b in client.bodies)
    assert client.health_calls == 3


def test_skips_unregistered_world_size():
    backend = FakeBackend(registered={("m", 2)})
    client = FakeClient()
    seqs = [seq("a", ws=2), seq("b", ws=4)]
    replayer = WarmupReplayer(seqs, backend, client, config())
    replayer.run()
    stats = replayer.stats()
    assert stats["replayed"] == 1
    assert stats["skipped_other_world_size"] == 1
    assert backend.submitted == ["a"]


def test_world_size_filter_from_config():
    backend = FakeBackend(registered={("m", 2), ("m", 4)})
    replayer = WarmupReplayer(
        [seq("a", ws=2), seq("b", ws=4)], backend, FakeClient(), config(world_size=4)
    )
    replayer.run()
    assert backend.submitted == ["b"]


def test_budget_limits_selection():
    backend = FakeBackend()
    client = FakeClient()
    seqs = [seq("a", ts=3), seq("b", ts=2), seq("c", ts=1)]
    replayer = WarmupReplayer(seqs, backend, client, config(max_warm_tokens=16))
    replayer.run()
    assert replayer.stats()["sequences_selected"] == 2
    assert sorted(backend.submitted) == ["a", "b"]


def test_ready_timeout():
    backend = FakeBackend(registered=())
    clock = FakeClock()
    replayer = WarmupReplayer(
        [seq("a")], backend, FakeClient(), config(ready_timeout_s=3), clock=clock
    )
    replayer.run()
    assert replayer.stats()["state"] == "timeout"
    assert backend.submitted == []


def test_prefetch_timeout_counts_pending_jobs():
    backend = FakeBackend(pending_polls=10**9)
    clock = FakeClock()
    replayer = WarmupReplayer(
        [seq("a")], backend, FakeClient(), config(prefetch_timeout_s=2), clock=clock
    )
    replayer.run()
    stats = replayer.stats()
    assert stats["prefetch_timeouts"] == 1
    assert stats["replayed"] == 0


def test_replay_failures_are_counted():
    client = FakeClient(fail={7})
    replayer = WarmupReplayer([seq("a")], FakeBackend(), client, config())
    replayer.run()
    stats = replayer.stats()
    assert stats["replay_failures"] == 1
    assert stats["replayed"] == 0


def test_prefetch_submit_errors_are_counted():
    class Raising(FakeBackend):
        def submit_prefetch(self, s):
            raise RuntimeError("no layout")

    replayer = WarmupReplayer([seq("a")], Raising(), FakeClient(), config())
    replayer.run()
    assert replayer.stats()["prefetch_errors"] == 1


def test_nothing_to_replay():
    replayer = WarmupReplayer([], FakeBackend(), FakeClient(), config())
    replayer.run()
    assert replayer.stats()["state"] == "done"


def test_stop_while_waiting():
    backend = FakeBackend(registered=())
    replayer = WarmupReplayer(
        [seq("a")], backend, FakeClient(), config(poll_interval_s=0.01)
    )
    replayer.start()
    replayer.stop()
    assert replayer.stats()["state"] == "stopped"


def test_warmup_config_parsing():
    cfg = WarmupConfig.from_dict(
        {
            "journal_dir": "/nas/j",
            "vllm_url": "http://127.0.0.1:8000",
            "max_warm_tokens": 100,
            "world_size": 4,
            "journal_retention_days": 7,
        }
    )
    assert cfg.replay_config().max_warm_tokens == 100
    assert cfg.replay_config().world_size == 4
    assert cfg.journal_retention_days == 7.0

    assert WarmupConfig.from_dict({"journal_dir": "/j", "replay": False}).vllm_url is None

    with pytest.raises(ValueError, match="vllm_url"):
        WarmupConfig.from_dict({"journal_dir": "/j"})
    with pytest.raises(ValueError, match="unknown"):
        WarmupConfig.from_dict({"journal_dir": "/j", "replay": False, "typo": 1})
    with pytest.raises(ValueError, match="replay_concurrency"):
        WarmupConfig.from_dict(
            {"journal_dir": "/j", "replay": False, "replay_concurrency": 0}
        )
    with pytest.raises(ValueError, match="journal_dir"):
        WarmupConfig.from_dict({"replay": False})
