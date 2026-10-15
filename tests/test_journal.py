# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# Third Party
import pytest

# First Party
from nas_kv_warmup.journal import (
    ChunkRecord,
    JournalWriter,
    build_sequences,
    compact_journal,
    load_journal,
    select_sequences,
    world_size_from_kv_rank,
)

CHUNK = 4


def rec(h, p, o, ts=100.0, model="m", ws=2, salt="", base=None):
    first = o if base is None else base
    return ChunkRecord(
        model=model,
        ws=ws,
        salt=salt,
        h=h,
        p=p,
        o=o,
        t=tuple(range(first, first + CHUNK)),
        ts=ts,
    )


def test_world_size_from_kv_rank():
    kv_rank = (8 << 24) | (5 << 16) | (4 << 8) | 1
    assert world_size_from_kv_rank(kv_rank) == 8


def test_record_json_roundtrip():
    record = rec("aa", "bb", 4, salt="tenant")
    assert ChunkRecord.from_json(record.to_json()) == record
    root = rec("aa", None, 0)
    assert ChunkRecord.from_json(root.to_json()) == root


def test_from_json_rejects_garbage():
    with pytest.raises(ValueError):
        ChunkRecord.from_json('{"model": "m"}')
    with pytest.raises(ValueError):
        ChunkRecord.from_json("{not json")


def test_writer_and_load_dedup_and_torn_lines(tmp_path):
    w1 = JournalWriter(tmp_path, "a.jsonl")
    w2 = JournalWriter(tmp_path, "b.jsonl")
    w1.append([rec("aa", None, 0, ts=1.0), rec("bb", "aa", 4, ts=1.0)])
    w2.append([rec("aa", None, 0, ts=5.0)])
    with open(w2.path, "a") as f:
        f.write('{"model": "m", "ws": 2, "h"')

    snapshot = load_journal(tmp_path)
    assert snapshot.skipped_lines == 1
    records = {r.h: r for r in snapshot.records()}
    assert set(records) == {"aa", "bb"}
    assert records["aa"].ts == 5.0


def test_load_retention(tmp_path):
    JournalWriter(tmp_path, "a.jsonl").append(
        [rec("old", None, 0, ts=10.0), rec("new", None, 0, ts=1000.0)]
    )
    snapshot = load_journal(tmp_path, retention_s=100, now=1050.0)
    assert [r.h for r in snapshot.records()] == ["new"]


def test_load_missing_dir(tmp_path):
    assert load_journal(tmp_path / "nope").records() == []


def test_build_sequences_chain_and_branches():
    records = [
        rec("r", None, 0, ts=1),
        rec("a", "r", 4, ts=2),
        rec("b", "a", 8, ts=3),
        rec("c", "r", 4, ts=4, base=100),
    ]
    seqs = {s.chunk_hashes: s for s in build_sequences(records)}
    assert set(seqs) == {("r", "a", "b"), ("r", "c")}
    long_seq = seqs[("r", "a", "b")]
    assert long_seq.token_ids == tuple(range(12))
    assert long_seq.ts == 3
    assert seqs[("r", "c")].token_ids == (0, 1, 2, 3, 100, 101, 102, 103)


def test_build_sequences_skips_broken_chains():
    records = [
        rec("orphan", "missing", 4),
        rec("not_root", None, 8),
        rec("r", None, 0),
        rec("gap", "r", 12),
        rec("x", "y", 4),
        rec("y", "x", 0),
    ]
    assert build_sequences(records) == []


def test_build_sequences_groups_by_world_size_and_salt():
    records = [
        rec("r", None, 0, ws=2),
        rec("r", None, 0, ws=4),
        rec("r", None, 0, salt="t1"),
    ]
    seqs = build_sequences(records)
    assert sorted((s.ws, s.salt) for s in seqs) == [(2, ""), (2, "t1"), (4, "")]


def test_select_sequences_budget_counts_shared_prefix_once():
    records = [
        rec("r", None, 0, ts=1),
        rec("a", "r", 4, ts=3),
        rec("c", "r", 4, ts=2, base=100),
        rec("z", None, 0, ts=0, base=500),
    ]
    seqs = build_sequences(records)

    assert len(select_sequences(seqs, 0)) == 3

    picked = select_sequences(seqs, 3 * CHUNK)
    assert [s.chunk_hashes for s in picked] == [("r", "a"), ("r", "c")]

    picked = select_sequences(seqs, 3 * CHUNK - 1)
    assert [s.chunk_hashes for s in picked] == [("r", "a")]


def test_compact_merges_idle_files_only(tmp_path):
    JournalWriter(tmp_path, "idle1.jsonl").append([rec("aa", None, 0, ts=1.0)])
    JournalWriter(tmp_path, "idle2.jsonl").append(
        [rec("aa", None, 0, ts=2.0), rec("bb", "aa", 4, ts=2.0)]
    )
    JournalWriter(tmp_path, "live.jsonl").append([rec("cc", None, 0, base=9)])
    os.utime(tmp_path / "idle1.jsonl", (0, 0))
    os.utime(tmp_path / "idle2.jsonl", (0, 0))
    now = (tmp_path / "live.jsonl").stat().st_mtime + 10

    snapshot = load_journal(tmp_path)
    merged = compact_journal(tmp_path, snapshot, min_idle_s=3600, now=now)

    assert merged == 2
    names = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert "live.jsonl" in names and "idle1.jsonl" not in names
    assert len(names) == 2
    reloaded = load_journal(tmp_path)
    assert sorted(r.h for r in reloaded.records()) == ["aa", "bb", "cc"]
    assert {r.h for r in snapshot.records()} == {"aa", "bb", "cc"}


def test_compact_noop_for_single_clean_file(tmp_path):
    JournalWriter(tmp_path, "a.jsonl").append([rec("aa", None, 0)])
    os.utime(tmp_path / "a.jsonl", (0, 0))
    snapshot = load_journal(tmp_path)
    assert compact_journal(tmp_path, snapshot, min_idle_s=1, now=10_000) == 0
    assert [p.name for p in tmp_path.glob("*.jsonl")] == ["a.jsonl"]


def test_compact_drops_expired_records(tmp_path):
    JournalWriter(tmp_path, "a.jsonl").append(
        [rec("old", None, 0, ts=1.0), rec("new", None, 0, ts=5000.0)]
    )
    os.utime(tmp_path / "a.jsonl", (0, 0))
    snapshot = load_journal(tmp_path, retention_s=100, now=5050.0)
    assert compact_journal(tmp_path, snapshot, min_idle_s=1, now=5050.0) == 1
    assert [r.h for r in load_journal(tmp_path).records()] == ["new"]
