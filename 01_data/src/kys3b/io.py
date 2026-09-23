"""Atomic IO, the memory-bounded document shuffle, and the log/stop contract.

`atomic_write_table` and `bucketed_shuffle` are ported from the 1.5B
`10_postprocess/pp_io.py`.  `bucketed_shuffle` in particular is the shuffle-parity
contract: seed 42, two passes, `default_rng(42)` over inputs in sorted order for the
scatter and `default_rng([42, bucket])` within each bucket for the gather.  Changing it
changes the training corpus.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- log / stop


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class Stop(RuntimeError):
    """A violated invariant.  Never caught inside the pipeline."""


def stop(msg: str) -> "None":
    """Abort with a non-zero exit.  The 1.5B `stop()` contract: never warn-and-continue."""
    print(f"\n*** STOP: {msg}\n", flush=True)
    raise Stop(msg)


def check(cond: bool, msg: str) -> None:
    if not cond:
        stop(msg)


# --------------------------------------------------------------------------- atomic writes


def atomic_write_table(table: pa.Table, dest: Path, compression: str = "zstd") -> None:
    """Write `table` to `dest` via `<dest>.tmp` in the SAME directory, then os.replace.

    On any exception the partial .tmp is unlinked, so a killed job never leaves a stale
    temp file behind.  Ported from pp_io.atomic_write_table.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        pq.write_table(table, tmp, compression=compression)
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(text: str, dest: Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(obj, dest: Path) -> None:
    atomic_write_text(json.dumps(obj, indent=2, sort_keys=False, default=_json_default), dest)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def atomic_save_npy(arr: np.ndarray, dest: Path) -> None:
    """Atomically write `arr` to `dest` (which must end in .npy).

    `np.save(path, arr)` APPENDS ".npy" when the path does not already end in it, so passing a
    temporary path like `doc_ids.npy.tmp` makes numpy write `doc_ids.npy.tmp.npy` and the
    subsequent os.replace then fails on a missing file, leaving a stray artifact behind.  Writing
    through an explicit file handle bypasses that rewriting entirely.  The handle is fsynced
    before the rename so a node failure cannot leave a renamed-but-empty file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, arr)          # explicit handle -> numpy does NOT touch the filename
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def parquet_rows(path: Path) -> int:
    """Row count from the footer only -- no data read."""
    return pq.ParquetFile(str(path)).metadata.num_rows


# --------------------------------------------------------------------------- cross-pass align


def paired_wiki_status(wiki: dict, distill: dict) -> np.ndarray:
    """Align the pass-1 `status` array to the distill row order.

    Both passes cover the SAME doc_ids in the same per-shard order, so equality is the
    fast path.  If it does not hold we fall back to an explicit doc_id join and raise if
    any id is missing -- never silently proceed.  Ported from pp_io.paired_wiki_status.
    """
    w_ids, d_ids = wiki["doc_id"], distill["doc_id"]
    if w_ids.shape == d_ids.shape and np.array_equal(w_ids, d_ids):
        return wiki["status"]
    order = np.argsort(w_ids, kind="stable")
    pos = np.searchsorted(w_ids[order], d_ids)
    check(bool((pos < w_ids.size).all()), "paired_wiki_status: distill doc_id beyond wiki range")
    idx = order[np.clip(pos, 0, w_ids.size - 1)]
    check(bool(np.array_equal(w_ids[idx], d_ids)), "paired_wiki_status: doc_id join failed")
    return wiki["status"][idx]


# --------------------------------------------------------------------------- shuffle


def bucketed_shuffle(
    in_paths: list[Path],
    out_dir: Path,
    tmp_dir: Path,
    seed: int = 42,
    rows_per_out_shard: int = 500_000,
    mem_gb: float = 240.0,
    n_buckets: int | None = None,
) -> dict:
    """Memory-bounded two-pass document-level shuffle.  Ported from pp_io.bucketed_shuffle.

    pass 1 (scatter): every input shard's rows go to one of B on-disk bucket files, bucket
                      id drawn from a single `default_rng(seed)` over the inputs in sorted
                      order.
    pass 2 (gather):  each bucket is read, shuffled within by `default_rng([seed, b])`, and
                      emitted as ~`rows_per_out_shard`-row parts with a global counter; a
                      carry buffer keeps the output shards uniform.

    Row count is asserted conserved.  Bucket temporaries are deleted as they are consumed.
    """
    in_paths = sorted(Path(p) for p in in_paths)
    out_dir, tmp_dir = Path(out_dir), Path(tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if n_buckets is None:
        on_disk = sum(p.stat().st_size for p in in_paths)
        text_bytes_est = on_disk * 4  # zstd -> Arrow, the 1.5B sizing model
        n_buckets = max(16, int(np.ceil(2.0 * text_bytes_est / (0.55 * mem_gb * 2**30))))
    rng = np.random.default_rng(seed)

    # ---- pass 1: scatter ----
    writers: dict[int, pq.ParquetWriter] = {}
    bucket_paths = {b: tmp_dir / f"bucket_{b:05d}.parquet" for b in range(n_buckets)}
    rows_in = 0
    schema = None
    try:
        for p in in_paths:
            t = pq.read_table(p, use_threads=False)
            rows_in += t.num_rows
            if schema is None:
                schema = t.schema
            bids = rng.integers(0, n_buckets, size=t.num_rows)
            for b in np.unique(bids):
                sub = t.filter(pa.array(bids == b))
                if sub.num_rows == 0:
                    continue
                b = int(b)
                if b not in writers:
                    writers[b] = pq.ParquetWriter(bucket_paths[b], schema, compression="zstd")
                writers[b].write_table(sub)
            del t
    finally:
        for w in writers.values():
            w.close()

    # ---- pass 2: gather ----
    carry: list[pa.Table] = []
    carry_rows = 0
    out_idx = 0
    rows_out = 0

    def _flush(force: bool) -> None:
        nonlocal carry, carry_rows, out_idx, rows_out
        while carry_rows >= rows_per_out_shard or (force and carry_rows > 0):
            merged = pa.concat_tables(carry)
            take = min(rows_per_out_shard, merged.num_rows)
            atomic_write_table(merged.slice(0, take), out_dir / f"part_{out_idx:05d}.parquet")
            rows_out += take
            out_idx += 1
            rest = merged.slice(take)
            carry = [rest] if rest.num_rows else []
            carry_rows = rest.num_rows
            if not force and carry_rows < rows_per_out_shard:
                break

    for b in range(n_buckets):
        bp = bucket_paths[b]
        if not bp.exists():
            continue
        t = pq.read_table(bp, use_threads=False)
        perm = np.random.default_rng([seed, b]).permutation(t.num_rows)
        t = t.take(pa.array(perm))
        carry.append(t)
        carry_rows += t.num_rows
        _flush(force=False)
        bp.unlink()
    _flush(force=True)

    check(rows_in == rows_out, f"bucketed_shuffle: rows_in {rows_in} != rows_out {rows_out}")
    return dict(n_buckets=n_buckets, rows=rows_in, out_shards=out_idx, seed=seed)
