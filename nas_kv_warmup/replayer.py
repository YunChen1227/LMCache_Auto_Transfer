# SPDX-License-Identifier: Apache-2.0
"""Startup warmup: replay journaled token prefixes into vLLM.

For each batch of selected sequences the replayer

1. submits an L2 -> L1 warm prefetch (NAS -> CPU) per sequence and keeps only
   sequences whose every key was found, so nothing that would need a real
   prefill is ever replayed;
2. sends each surviving sequence to vLLM as ``/v1/completions`` with the token
   ids as ``prompt`` and ``max_tokens=1``. vLLM's lookup hits LMCache L1, the
   KV is loaded into GPU blocks, and vLLM keeps them in its prefix cache.

Batches bound how much L1 the warmup holds at once. LMCache access goes through
the small :class:`WarmupBackend` protocol so this module stays importable
without LMCache.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Protocol
import json
import logging
import threading
import time
import urllib.error
import urllib.request

# First Party
from nas_kv_warmup.journal import WarmSequence, select_sequences

logger = logging.getLogger(__name__)


class WarmupBackend(Protocol):
    """The LMCache operations the replayer needs."""

    def is_registered(self, model: str, world_size: int) -> bool:
        """Whether vLLM has registered its KV cache for this pair."""
        ...

    def submit_prefetch(self, seq: WarmSequence) -> Optional[str]:
        """Start an L2 -> L1 warm prefetch; ``None`` when there is nothing to load."""
        ...

    def poll_prefetch(self, job_id: str) -> Optional[tuple[int, int]]:
        """``None`` while pending, else ``(found_keys, total_keys)``."""
        ...


class VLLMClient:
    """Minimal OpenAI-compatible client for the vLLM server."""

    def __init__(self, base_url: str, request_timeout_s: float = 600.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = request_timeout_s

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self._base_url}/health", timeout=5.0
            ) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def complete(self, body: dict) -> None:
        """POST ``/v1/completions``; raises on transport or HTTP errors."""
        request = urllib.request.Request(
            f"{self._base_url}/v1/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as resp:
            resp.read()


@dataclass(frozen=True)
class ReplayConfig:
    """Replay settings (see the README for the matching ``config`` keys)."""

    served_model_name: Optional[str] = None
    max_warm_tokens: int = 0
    replay_concurrency: int = 8
    ready_timeout_s: float = 1800.0
    prefetch_batch_size: int = 32
    prefetch_timeout_s: float = 600.0
    poll_interval_s: float = 1.0
    world_size: Optional[int] = None


def build_completion_body(seq: WarmSequence, served_model_name: Optional[str]) -> dict:
    body: dict = {
        "model": served_model_name or seq.model,
        "prompt": list(seq.token_ids),
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    if seq.salt:
        body["cache_salt"] = seq.salt
    return body


class WarmupReplayer:
    """Runs the startup warmup once, on a background thread or inline."""

    def __init__(
        self,
        sequences: Sequence[WarmSequence],
        backend: WarmupBackend,
        client: VLLMClient,
        config: ReplayConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sequences = list(sequences)
        self._backend = backend
        self._client = client
        self._config = config
        self._clock = clock

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stats = {
            "state": "idle",
            "sequences_total": len(self._sequences),
            "sequences_selected": 0,
            "skipped_other_world_size": 0,
            "skipped_missing": 0,
            "prefetch_timeouts": 0,
            "prefetch_errors": 0,
            "replayed": 0,
            "replay_failures": 0,
            "tokens_warmed": 0,
            "elapsed_s": 0.0,
        }

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.run, name="nas-kv-warmup-replayer", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _set(self, **updates) -> None:
        with self._lock:
            self._stats.update(updates)

    def _add(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._stats[name] += value

    def run(self) -> None:
        started = self._clock()
        try:
            self._run(started)
        except Exception:
            logger.exception("NAS KV warmup failed")
            self._set(state="failed")
        finally:
            self._set(elapsed_s=round(self._clock() - started, 3))

    def _run(self, started: float) -> None:
        cfg = self._config
        candidates = self._sequences
        if cfg.world_size is not None:
            candidates = [s for s in candidates if s.ws == cfg.world_size]
        if not candidates:
            self._set(state="done")
            logger.info("NAS KV warmup: nothing to replay")
            return

        self._set(state="waiting")
        logger.info(
            "NAS KV warmup: %d sequences loaded, waiting for vLLM to register "
            "its KV cache and pass /health",
            len(candidates),
        )
        ready_pairs = self._wait_until_ready({(s.model, s.ws) for s in candidates})
        if ready_pairs is None:
            if not self._stop.is_set():
                logger.warning(
                    "NAS KV warmup: vLLM / KV layout not ready within %.0fs, giving up",
                    cfg.ready_timeout_s,
                )
                self._set(state="timeout")
            else:
                self._set(state="stopped")
            return

        replayable = [s for s in candidates if (s.model, s.ws) in ready_pairs]
        self._set(skipped_other_world_size=len(self._sequences) - len(replayable))
        selected = select_sequences(replayable, cfg.max_warm_tokens)
        self._set(state="running", sequences_selected=len(selected))
        logger.info(
            "NAS KV warmup: replaying %d of %d sequences (%d tokens)",
            len(selected),
            len(self._sequences),
            sum(s.num_tokens for s in selected),
        )

        # Oldest first, so the newest prefixes are the most recently used in
        # vLLM's prefix cache and the last to be evicted.
        ordered = list(reversed(selected))
        batch_size = max(1, cfg.prefetch_batch_size)
        with ThreadPoolExecutor(
            max_workers=max(1, cfg.replay_concurrency),
            thread_name_prefix="nas-kv-warmup-req",
        ) as pool:
            for start in range(0, len(ordered), batch_size):
                if self._stop.is_set():
                    self._set(state="stopped")
                    return
                batch = ordered[start : start + batch_size]
                ready = self._prefetch_batch(batch)
                list(pool.map(self._replay_one, ready))
                stats = self.stats()
                logger.info(
                    "NAS KV warmup progress: %d/%d sequences processed, "
                    "replayed=%d skipped_missing=%d tokens_warmed=%d",
                    min(start + batch_size, len(ordered)),
                    len(ordered),
                    stats["replayed"],
                    stats["skipped_missing"],
                    stats["tokens_warmed"],
                )
        self._set(state="done", elapsed_s=round(self._clock() - started, 3))
        logger.info("NAS KV warmup finished: %s", self.stats())

    def _wait_until_ready(
        self, pairs: set[tuple[str, int]]
    ) -> Optional[set[tuple[str, int]]]:
        deadline = self._clock() + self._config.ready_timeout_s
        while not self._stop.is_set():
            ready = {p for p in pairs if self._backend.is_registered(*p)}
            if ready and self._client.health():
                return ready
            if self._clock() >= deadline:
                return None
            self._stop.wait(self._config.poll_interval_s)
        return None

    def _prefetch_batch(self, batch: Sequence[WarmSequence]) -> list[WarmSequence]:
        jobs: dict[str, WarmSequence] = {}
        for seq in batch:
            try:
                job_id = self._backend.submit_prefetch(seq)
            except Exception as exc:
                logger.warning("NAS KV warmup: prefetch submit failed: %s", exc)
                self._add("prefetch_errors")
                continue
            if job_id is None:
                self._add("skipped_missing")
                continue
            jobs[job_id] = seq

        ready: list[WarmSequence] = []
        deadline = self._clock() + self._config.prefetch_timeout_s
        while jobs and not self._stop.is_set():
            for job_id in list(jobs):
                result = self._backend.poll_prefetch(job_id)
                if result is None:
                    continue
                seq = jobs.pop(job_id)
                found, total = result
                if total > 0 and found == total:
                    ready.append(seq)
                else:
                    self._add("skipped_missing")
            if not jobs or self._clock() >= deadline:
                break
            self._stop.wait(min(0.05, self._config.poll_interval_s))
        if jobs:
            self._add("prefetch_timeouts", len(jobs))
        return ready

    def _replay_one(self, seq: WarmSequence) -> None:
        if self._stop.is_set():
            return
        body = build_completion_body(seq, self._config.served_model_name)
        try:
            self._client.complete(body)
        except Exception as exc:
            logger.warning(
                "NAS KV warmup: replay of %d tokens failed: %s", seq.num_tokens, exc
            )
            self._add("replay_failures")
            return
        self._add("replayed")
        self._add("tokens_warmed", seq.num_tokens)
