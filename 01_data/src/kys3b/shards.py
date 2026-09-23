"""Materialize per-setting rewriting input: 10,000-row (doc_id, text) parquet shards.

`_shards.json` is the FINGERPRINT of the shard geometry.  It pins shard_index, and
shard_index is what seeds the WRAP style assignment, so it must not drift under a rerun.

Reading text out of the pool is done shard-by-shard over the POOL's 200 files with the
selected doc_ids grouped by pool shard, so each pool shard is opened exactly once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .io import atomic_write_json, atomic_write_table, check, log, parquet_rows
from .pool import doc_id_to_shard

TEXT_COLS = ["doc_id", "text"]


def shard_path(out_dir: Path, i: int) -> Path:
    return Path(out_dir) / f"shard_{i:05d}.parquet"


def plan_shards(doc_ids: np.ndarray, rows_per_shard: int) -> list[np.ndarray]:
    """Split the SORTED doc_id array into fixed-size shards.

    Sorting first makes the geometry a pure function of the doc_id set, so `_shards.json`
    is reproducible and `shard_index` cannot drift between a report and a commit run.
    """
    d = np.sort(np.asarray(doc_ids, dtype=np.int64))
    n = d.size
    return [d[i : i + rows_per_shard] for i in range(0, n, rows_per_shard)]


def materialize(
    cfg,
    setting: str,
    doc_ids: np.ndarray,
    out_dir: Path,
    rows_per_shard: int,
    batch_shards: int = 64,
) -> dict:
    """Write the (doc_id, text) shards for one setting.  Idempotent per shard.

    Output shards are processed in batches of `batch_shards` so peak memory is bounded.  This
    matters at production scale: quality-first is ~12.2M documents over ~1,220 shards, and
    buffering all of their text at once would be ~80 GB.  A 64-shard batch is ~640k documents,
    a couple of GB of text.

    Because `doc_ids` is sorted, each output shard covers a narrow contiguous doc_id range and
    therefore touches only one or two pool shards, so batching costs almost no extra pool reads.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = plan_shards(doc_ids, rows_per_shard)
    log(f"{setting}: {len(doc_ids):,} docs -> {len(groups)} shards of <= {rows_per_shard}")

    todo = []
    for si, ids in enumerate(groups):
        p = shard_path(out_dir, si)
        if p.exists() and parquet_rows(p) == ids.size:
            continue
        todo.append(si)
    if not todo:
        log(f"{setting}: all {len(groups)} shards already materialized")
    else:
        log(f"{setting}: {len(todo)} shards to write, in batches of {batch_shards}")

    written = 0
    for b0 in range(0, len(todo), batch_shards):
        batch = todo[b0 : b0 + batch_shards]
        # group the needed (out_shard, rows) by pool shard so each pool file is read once per batch
        need: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for si in batch:
            ids = groups[si]
            psh, prow = doc_id_to_shard(cfg, ids)
            for ps in np.unique(psh):
                m = psh == ps
                need.setdefault(int(ps), []).append((si, prow[m], ids[m]))

        buf: dict[int, dict[int, str]] = {si: {} for si in batch}
        for ps in sorted(need):
            t = pq.read_table(cfg.pool_shard(ps), columns=TEXT_COLS, use_threads=False)
            pool_ids = t.column("doc_id").to_numpy(zero_copy_only=False)
            texts = t.column("text").to_pylist()
            for si, prow, ids in need[ps]:
                check(
                    bool(np.array_equal(pool_ids[prow], ids)),
                    f"{setting} shard {si}: doc_id mismatch reading pool shard {ps}",
                )
                for did, r in zip(ids.tolist(), prow.tolist()):
                    buf[si][did] = texts[r]
            del t, texts, pool_ids

        for si in batch:
            ids = groups[si]
            tbl = pa.table(
                {
                    "doc_id": pa.array(ids, type=pa.int64()),
                    "text": pa.array([buf[si][int(d)] for d in ids], type=pa.large_string()),
                }
            )
            atomic_write_table(tbl, shard_path(out_dir, si))
            written += 1
            buf[si] = {}
        del buf, need
        log(f"{setting}: {written}/{len(todo)} shards written")

    index = dict(
        setting=setting,
        rows_per_shard=rows_per_shard,
        n_shards=len(groups),
        total_docs=int(len(doc_ids)),
        shards=[
            dict(
                shard=si,
                rows=int(g.size),
                doc_id_min=int(g[0]),
                doc_id_max=int(g[-1]),
                file=shard_path(out_dir, si).name,
            )
            for si, g in enumerate(groups)
        ],
    )
    atomic_write_json(index, out_dir / "_shards.json")
    log(f"{setting}: index -> {out_dir/'_shards.json'}")
    return index


def load_index(out_dir: Path) -> dict:
    import json

    p = Path(out_dir) / "_shards.json"
    check(p.exists(), f"missing {p} -- run bin/02_materialize_sources.py first")
    return json.loads(p.read_text())
