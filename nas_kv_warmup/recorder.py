# SPDX-License-Identifier: Apache-2.0
"""Event-bus subscriber that writes the token journal.

``MP_TOKENS`` binds each chunk hash to its tokens when a store is submitted;
an ``L1_WRITE_FINISHED*`` event later confirms the chunk actually landed and
carries the ``ObjectKey`` fields (model, kv rank, cache salt). A journal record
is emitted only once both are seen, mirroring ``KVEventModule``.

Event callbacks run on the event bus drain thread, so they only touch
in-memory state and hand records to a background writer thread.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional
import logging
import queue
import threading
import time

# First Party
from nas_kv_warmup.journal import (
    ChunkRecord,
    JournalWriter,
    RecordKey,
    world_size_from_kv_rank,
)

logger = logging.getLogger(__name__)

_Binding = tuple[tuple[int, ...], Optional[bytes], int]


class JournalRecorder:
    """Turn store events into journal records.

    Args:
        writer: Destination journal file.
        tokens_event_type: The ``EventType.MP_TOKENS`` member.
        write_event_types: The ``EventType.L1_WRITE_FINISHED*`` members.
        binding_cache_size: Max chunk-hash -> tokens bindings kept in memory
            while waiting for the matching write-finished event.
        seen_cache_size: Max record keys remembered for deduplication.
        queue_size: Max records buffered for the writer thread; overflow is
            dropped and counted.
        flush_interval_s: Max seconds a record waits before being flushed.
        seen: Record keys already in the journal (e.g. from earlier runs).
        clock: Wall-clock source, for tests.
    """

    def __init__(
        self,
        writer: JournalWriter,
        tokens_event_type: Any,
        write_event_types: Sequence[Any],
        binding_cache_size: int = 65536,
        seen_cache_size: int = 1_000_000,
        queue_size: int = 65536,
        flush_interval_s: float = 1.0,
        seen: Iterable[RecordKey] = (),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._writer = writer
        self._tokens_type = tokens_event_type
        self._write_types = frozenset(write_event_types)
        self._binding_cache_size = binding_cache_size
        self._seen_cache_size = seen_cache_size
        self._flush_interval_s = flush_interval_s
        self._clock = clock

        self._lock = threading.Lock()
        self._bindings: OrderedDict[bytes, _Binding] = OrderedDict()
        self._seen: OrderedDict[RecordKey, None] = OrderedDict(
            (key, None) for key in seen
        )
        self._trim_seen()
        self._queue: queue.Queue[ChunkRecord] = queue.Queue(maxsize=queue_size)

        self._recorded = 0
        self._unbound = 0
        self._dropped = 0
        self._write_failures = 0

        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="nas-kv-journal-writer", daemon=True
        )
        self._thread.start()

    @property
    def path(self) -> Any:
        return self._writer.path

    def attach(self, event_bus: Any) -> None:
        """Subscribe to the events this recorder needs."""
        event_bus.subscribe(self._tokens_type, self.on_event)
        for event_type in self._write_types:
            event_bus.subscribe(event_type, self.on_event)

    def on_event(self, event: Any) -> None:
        if self._closed.is_set():
            return
        if event.event_type == self._tokens_type:
            self._on_tokens(event.metadata)
        elif event.event_type in self._write_types:
            self._on_write_finished(event.metadata)

    def _on_tokens(self, metadata: dict) -> None:
        chunk_hashes = metadata.get("chunk_hashes", [])
        token_chunks = metadata.get("token_chunks", [])
        parent_hashes = metadata.get("parent_hashes", [])
        token_offsets = metadata.get("token_offsets", [])
        if not (
            len(chunk_hashes)
            == len(token_chunks)
            == len(parent_hashes)
            == len(token_offsets)
        ):
            return
        with self._lock:
            for chunk_hash, tokens, parent, offset in zip(
                chunk_hashes, token_chunks, parent_hashes, token_offsets
            ):
                self._bindings[chunk_hash] = (tuple(tokens), parent, int(offset))
                self._bindings.move_to_end(chunk_hash)
            while len(self._bindings) > self._binding_cache_size:
                self._bindings.popitem(last=False)

    def _on_write_finished(self, metadata: dict) -> None:
        keys = metadata.get("keys", [])
        now = self._clock()
        with self._lock:
            for key in keys:
                record_key: RecordKey = (
                    key.model_name,
                    world_size_from_kv_rank(key.kv_rank),
                    key.cache_salt,
                    key.chunk_hash.hex(),
                )
                if record_key in self._seen:
                    continue
                binding = self._bindings.get(key.chunk_hash)
                if binding is None:
                    self._unbound += 1
                    continue
                tokens, parent, offset = binding
                record = ChunkRecord(
                    model=record_key[0],
                    ws=record_key[1],
                    salt=record_key[2],
                    h=record_key[3],
                    p=None if parent is None else parent.hex(),
                    o=offset,
                    t=tokens,
                    ts=now,
                )
                try:
                    self._queue.put_nowait(record)
                except queue.Full:
                    self._dropped += 1
                    continue
                self._seen[record_key] = None
                self._recorded += 1
            self._trim_seen()

    def _trim_seen(self) -> None:
        while len(self._seen) > self._seen_cache_size:
            self._seen.popitem(last=False)

    def _run(self) -> None:
        while not self._closed.is_set():
            self._flush_once(block=True)
        self._flush_once(block=False)

    def _flush_once(self, block: bool) -> None:
        batch: list[ChunkRecord] = []
        try:
            if block:
                batch.append(self._queue.get(timeout=self._flush_interval_s))
            while True:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        if not batch:
            return
        try:
            self._writer.append(batch)
        except OSError as exc:
            with self._lock:
                self._write_failures += 1
            logger.warning(
                "Failed to append %d records to journal %s: %s",
                len(batch),
                self._writer.path,
                exc,
            )

    def stats(self) -> dict:
        with self._lock:
            return {
                "journal_file": str(self._writer.path),
                "recorded": self._recorded,
                "unbound_writes": self._unbound,
                "dropped": self._dropped,
                "write_failures": self._write_failures,
                "pending": self._queue.qsize(),
            }

    def close(self, timeout: float = 10.0) -> None:
        """Stop accepting events and flush what is buffered."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._thread.join(timeout=timeout)
