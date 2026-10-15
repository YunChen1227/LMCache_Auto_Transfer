# SPDX-License-Identifier: Apache-2.0
"""Token journal persisted next to the NAS KV files.

Every KV chunk LMCache writes is identified on NAS only by its chunk hash, so
the tokens behind it cannot be recovered from the KV files. The journal keeps
one JSON line per stored chunk::

    {"model": ..., "ws": 4, "salt": "", "h": "<hex>", "p": "<hex>|null",
     "o": 0, "t": [token ids], "ts": 1700000000.0}

``h`` / ``p`` are the chunk hash and its parent chunk hash, ``o`` is the
absolute offset of the chunk's first token and ``ws`` the world size the KV
was stored under. Chains of chunks rebuild the full token prefixes that can be
replayed into vLLM.

Each server instance appends to its own file so concurrent writers never share
a file on NAS. This module only depends on the standard library.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import logging
import os
import socket
import time
import uuid

logger = logging.getLogger(__name__)

JOURNAL_SUFFIX = ".jsonl"

RecordKey = tuple[str, int, str, str]


def world_size_from_kv_rank(kv_rank: int) -> int:
    """Decode the world size packed by ``ObjectKey.ComputeKVRank``."""
    return (kv_rank >> 24) & 0xFF


@dataclass(frozen=True)
class ChunkRecord:
    """One stored KV chunk and the tokens it was computed from."""

    model: str
    ws: int
    salt: str
    h: str
    p: Optional[str]
    o: int
    t: tuple[int, ...]
    ts: float

    @property
    def key(self) -> RecordKey:
        return (self.model, self.ws, self.salt, self.h)

    def to_json(self) -> str:
        return json.dumps(
            {
                "model": self.model,
                "ws": self.ws,
                "salt": self.salt,
                "h": self.h,
                "p": self.p,
                "o": self.o,
                "t": list(self.t),
                "ts": self.ts,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> "ChunkRecord":
        """Parse one journal line.

        Raises:
            ValueError: The line is not a well-formed record.
        """
        try:
            d = json.loads(line)
            parent = d.get("p")
            return cls(
                model=str(d["model"]),
                ws=int(d["ws"]),
                salt=str(d.get("salt", "")),
                h=str(d["h"]),
                p=None if parent is None else str(parent),
                o=int(d["o"]),
                t=tuple(int(x) for x in d["t"]),
                ts=float(d["ts"]),
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed journal record: {exc}") from None


@dataclass(frozen=True)
class WarmSequence:
    """A full token prefix rebuilt from a chain of journal records."""

    model: str
    ws: int
    salt: str
    token_ids: tuple[int, ...]
    chunk_hashes: tuple[str, ...]
    ts: float

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class JournalSnapshot:
    """Records loaded from a journal directory, grouped by source file."""

    files: dict[Path, list[ChunkRecord]]
    line_counts: dict[Path, int] = field(default_factory=dict)
    skipped_lines: int = 0

    def records(self) -> list[ChunkRecord]:
        """All records across files, deduplicated (newest ``ts`` wins)."""
        return dedup_records(r for recs in self.files.values() for r in recs)


def dedup_records(records: Iterable[ChunkRecord]) -> list[ChunkRecord]:
    """Keep one record per ``(model, ws, salt, h)``, the one with newest ``ts``."""
    latest: dict[RecordKey, ChunkRecord] = {}
    for record in records:
        current = latest.get(record.key)
        if current is None or record.ts > current.ts:
            latest[record.key] = record
    return list(latest.values())


def default_journal_filename() -> str:
    return f"{socket.gethostname()}-{os.getpid()}{JOURNAL_SUFFIX}"


class JournalWriter:
    """Appends records to this instance's journal file."""

    def __init__(self, journal_dir: str | os.PathLike, filename: str | None = None):
        self._dir = Path(journal_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / (filename or default_journal_filename())

    @property
    def path(self) -> Path:
        return self._path

    def append(self, records: Sequence[ChunkRecord]) -> None:
        if not records:
            return
        data = "".join(record.to_json() + "\n" for record in records)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())


def load_journal(
    journal_dir: str | os.PathLike,
    retention_s: float = 0.0,
    now: float | None = None,
) -> JournalSnapshot:
    """Read every journal file under ``journal_dir``.

    Args:
        journal_dir: Directory holding ``*.jsonl`` journal files.
        retention_s: Drop records older than this many seconds; 0 keeps all.
        now: Current wall-clock time, for tests.
    """
    directory = Path(journal_dir)
    snapshot = JournalSnapshot(files={})
    if not directory.is_dir():
        return snapshot
    cutoff = (now if now is not None else time.time()) - retention_s
    for path in sorted(directory.glob(f"*{JOURNAL_SUFFIX}")):
        records: list[ChunkRecord] = []
        line_count = 0
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line_count += 1
                    try:
                        record = ChunkRecord.from_json(line)
                    except ValueError:
                        # A torn trailing line from a crashed writer.
                        snapshot.skipped_lines += 1
                        continue
                    if retention_s > 0 and record.ts < cutoff:
                        continue
                    records.append(record)
        except OSError as exc:
            logger.warning("Cannot read journal file %s: %s", path, exc)
            continue
        snapshot.files[path] = records
        snapshot.line_counts[path] = line_count
    return snapshot


def compact_journal(
    journal_dir: str | os.PathLike,
    snapshot: JournalSnapshot,
    min_idle_s: float,
    now: float | None = None,
) -> int:
    """Merge idle journal files into one deduplicated file.

    Only files whose mtime is older than ``min_idle_s`` are merged, so files
    that a live instance is still appending to are left untouched. Records
    already filtered out of ``snapshot`` (e.g. past retention) are dropped.

    Returns:
        The number of files merged (0 when nothing was compacted).
    """
    directory = Path(journal_dir)
    now = now if now is not None else time.time()
    idle: list[Path] = []
    for path in snapshot.files:
        try:
            if now - path.stat().st_mtime >= min_idle_s:
                idle.append(path)
        except OSError:
            continue
    if not idle:
        return 0

    merged = dedup_records(r for path in idle for r in snapshot.files[path])
    total_before = sum(snapshot.line_counts.get(path, 0) for path in idle)
    if len(idle) == 1 and len(merged) == total_before:
        return 0

    target = directory / f"compacted-{uuid.uuid4().hex}{JOURNAL_SUFFIX}"
    tmp = target.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in merged:
            f.write(record.to_json() + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    for path in idle:
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Cannot remove compacted journal file %s: %s", path, exc)
    for path in idle:
        snapshot.files.pop(path, None)
        snapshot.line_counts.pop(path, None)
    snapshot.files[target] = merged
    snapshot.line_counts[target] = len(merged)
    logger.info(
        "Compacted %d journal files into %s (%d -> %d records)",
        len(idle),
        target.name,
        total_before,
        len(merged),
    )
    return len(idle)


def build_sequences(records: Iterable[ChunkRecord]) -> list[WarmSequence]:
    """Rebuild full token prefixes by walking each leaf chunk back to its root.

    A chain is dropped when a parent is missing, the chain loops, the root does
    not start at offset 0, or chunk offsets are not contiguous.
    """
    groups: dict[tuple[str, int, str], dict[str, ChunkRecord]] = {}
    for record in dedup_records(records):
        groups.setdefault((record.model, record.ws, record.salt), {})[record.h] = record

    sequences: list[WarmSequence] = []
    for (model, ws, salt), nodes in groups.items():
        parents = {r.p for r in nodes.values() if r.p is not None}
        for leaf in nodes.values():
            if leaf.h in parents:
                continue
            chain = _walk_to_root(leaf, nodes)
            if chain is None:
                continue
            chain.reverse()
            token_ids: list[int] = []
            for record in chain:
                token_ids.extend(record.t)
            sequences.append(
                WarmSequence(
                    model=model,
                    ws=ws,
                    salt=salt,
                    token_ids=tuple(token_ids),
                    chunk_hashes=tuple(r.h for r in chain),
                    ts=max(r.ts for r in chain),
                )
            )
    return sequences


def _walk_to_root(
    leaf: ChunkRecord, nodes: dict[str, ChunkRecord]
) -> list[ChunkRecord] | None:
    chain = [leaf]
    seen = {leaf.h}
    current = leaf
    while current.p is not None:
        parent = nodes.get(current.p)
        if parent is None or parent.h in seen:
            return None
        if parent.o + len(parent.t) != current.o:
            return None
        chain.append(parent)
        seen.add(parent.h)
        current = parent
    if current.o != 0:
        return None
    return chain


def select_sequences(
    sequences: Iterable[WarmSequence], max_tokens: int = 0
) -> list[WarmSequence]:
    """Pick sequences newest first until the unique-token budget is used up.

    Chunks shared between sequences (common prefixes) are counted once, which
    matches how vLLM shares their GPU blocks. ``max_tokens <= 0`` means no
    budget. Sequences that do not fit are skipped so smaller ones can still
    fill the remaining budget.

    Returns:
        The selected sequences, newest first.
    """
    selected: list[WarmSequence] = []
    counted: set[RecordKey] = set()
    used = 0
    for seq in sorted(sequences, key=lambda s: s.ts, reverse=True):
        if not seq.chunk_hashes:
            continue
        chunk_tokens = len(seq.token_ids) // len(seq.chunk_hashes)
        new_keys = [
            (seq.model, seq.ws, seq.salt, h)
            for h in seq.chunk_hashes
            if (seq.model, seq.ws, seq.salt, h) not in counted
        ]
        cost = len(new_keys) * chunk_tokens
        if max_tokens > 0 and used + cost > max_tokens:
            continue
        counted.update(new_keys)
        used += cost
        selected.append(seq)
    return selected
