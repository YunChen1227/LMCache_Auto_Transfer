# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time

# First Party
from nas_kv_warmup.journal import JournalWriter, load_journal
from nas_kv_warmup.recorder import JournalRecorder


class FakeEventType(Enum):
    MP_TOKENS = "mp.tokens"
    L1_WRITE_FINISHED = "l1.write.finished"
    L1_WRITE_FINISHED_AND_READ_RESERVED = "l1.write_finished_and_read_reserved"
    OTHER = "other"


@dataclass
class FakeEvent:
    event_type: Any
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FakeKey:
    chunk_hash: bytes
    model_name: str
    kv_rank: int
    object_group_id: int = 0
    cache_salt: str = ""


class FakeBus:
    def __init__(self):
        self.subs = {}

    def subscribe(self, event_type, callback):
        self.subs.setdefault(event_type, []).append(callback)

    def publish(self, event):
        for cb in self.subs.get(event.event_type, []):
            cb(event)


def kv_rank(ws, rank):
    return (ws << 24) | (rank << 16) | (ws << 8) | rank


def make_recorder(tmp_path, **kwargs):
    recorder = JournalRecorder(
        JournalWriter(tmp_path, "j.jsonl"),
        FakeEventType.MP_TOKENS,
        (
            FakeEventType.L1_WRITE_FINISHED,
            FakeEventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
        ),
        flush_interval_s=0.05,
        clock=lambda: 42.0,
        **kwargs,
    )
    bus = FakeBus()
    recorder.attach(bus)
    return recorder, bus


def tokens_event(hashes, chunks, parents, offsets):
    return FakeEvent(
        FakeEventType.MP_TOKENS,
        {
            "chunk_hashes": hashes,
            "token_chunks": chunks,
            "parent_hashes": parents,
            "token_offsets": offsets,
        },
    )


def write_event(keys, event_type=FakeEventType.L1_WRITE_FINISHED):
    return FakeEvent(event_type, {"keys": keys})


def test_records_only_after_binding_and_write(tmp_path):
    recorder, bus = make_recorder(tmp_path)
    h1, h2 = b"\x01" * 8, b"\x02" * 8

    bus.publish(write_event([FakeKey(h1, "m", kv_rank(2, 0))]))
    bus.publish(tokens_event([h1, h2], [[1, 2], [3, 4]], [None, h1], [0, 2]))
    bus.publish(
        write_event(
            [
                FakeKey(h1, "m", kv_rank(2, 0)),
                FakeKey(h1, "m", kv_rank(2, 1)),
                FakeKey(h2, "m", kv_rank(2, 0)),
            ],
            FakeEventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
        )
    )
    recorder.close()

    stats = recorder.stats()
    assert stats["recorded"] == 2
    assert stats["unbound_writes"] == 1
    records = {r.h: r for r in load_journal(tmp_path).records()}
    assert set(records) == {h1.hex(), h2.hex()}
    r2 = records[h2.hex()]
    assert (r2.model, r2.ws, r2.p, r2.o, r2.t, r2.ts) == (
        "m",
        2,
        h1.hex(),
        2,
        (3, 4),
        42.0,
    )
    assert records[h1.hex()].p is None


def test_seen_keys_are_not_rewritten(tmp_path):
    h1 = b"\x01" * 8
    recorder, bus = make_recorder(tmp_path, seen=[("m", 2, "", h1.hex())])
    bus.publish(tokens_event([h1], [[1, 2]], [None], [0]))
    bus.publish(write_event([FakeKey(h1, "m", kv_rank(2, 0))]))
    recorder.close()
    assert recorder.stats()["recorded"] == 0
    assert load_journal(tmp_path).records() == []


def test_different_world_size_and_salt_are_separate_records(tmp_path):
    h1 = b"\x01" * 8
    recorder, bus = make_recorder(tmp_path)
    bus.publish(tokens_event([h1], [[1, 2]], [None], [0]))
    bus.publish(
        write_event(
            [
                FakeKey(h1, "m", kv_rank(2, 0)),
                FakeKey(h1, "m", kv_rank(4, 0)),
                FakeKey(h1, "m", kv_rank(2, 0), cache_salt="t"),
            ]
        )
    )
    recorder.close()
    keys = sorted((r.ws, r.salt) for r in load_journal(tmp_path).records())
    assert keys == [(2, ""), (2, "t"), (4, "")]


def test_binding_cache_is_bounded(tmp_path):
    recorder, bus = make_recorder(tmp_path, binding_cache_size=2)
    hashes = [bytes([i]) * 8 for i in range(3)]
    bus.publish(tokens_event(hashes, [[i] for i in range(3)], [None] * 3, [0] * 3))
    bus.publish(write_event([FakeKey(h, "m", kv_rank(1, 0)) for h in hashes]))
    recorder.close()
    stats = recorder.stats()
    assert stats["recorded"] == 2
    assert stats["unbound_writes"] == 1


def test_mismatched_tokens_metadata_is_ignored(tmp_path):
    h1 = b"\x01" * 8
    recorder, bus = make_recorder(tmp_path)
    bus.publish(tokens_event([h1], [[1], [2]], [None], [0]))
    bus.publish(write_event([FakeKey(h1, "m", kv_rank(1, 0))]))
    recorder.close()
    assert recorder.stats()["unbound_writes"] == 1


def test_events_after_close_are_ignored(tmp_path):
    h1 = b"\x01" * 8
    recorder, bus = make_recorder(tmp_path)
    recorder.close()
    bus.publish(tokens_event([h1], [[1]], [None], [0]))
    bus.publish(write_event([FakeKey(h1, "m", kv_rank(1, 0))]))
    assert recorder.stats()["recorded"] == 0


def test_writer_flushes_in_background(tmp_path):
    h1 = b"\x01" * 8
    recorder, bus = make_recorder(tmp_path)
    bus.publish(tokens_event([h1], [[1]], [None], [0]))
    bus.publish(write_event([FakeKey(h1, "m", kv_rank(1, 0))]))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not load_journal(tmp_path).records():
        time.sleep(0.02)
    assert len(load_journal(tmp_path).records()) == 1
    recorder.close()


def test_write_failures_are_counted(tmp_path):
    class BrokenWriter:
        path = tmp_path / "broken.jsonl"

        def append(self, records):
            raise OSError("NAS unavailable")

    recorder = JournalRecorder(
        BrokenWriter(),
        FakeEventType.MP_TOKENS,
        (FakeEventType.L1_WRITE_FINISHED,),
        flush_interval_s=0.05,
    )
    h1 = b"\x01" * 8
    recorder.on_event(tokens_event([h1], [[1]], [None], [0]))
    recorder.on_event(write_event([FakeKey(h1, "m", kv_rank(1, 0))]))
    recorder.close()
    assert recorder.stats()["write_failures"] == 1
