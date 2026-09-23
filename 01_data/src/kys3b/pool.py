"""The scored pool: shard geometry, the numeric-column load, and stage-00 validation.

Ported from 1.5B `04_select/select_10b.py` PASS1 (`_read_numeric` / `pass1`), including
every `stop()` gate: per-shard row count, doc_id contiguity, zero nulls, exactly 24 topics.

Columns used (all present in blab-jhu/KYS-DCLM-Refinedweb-100M-Scored):
    doc_id, tokens-llama2, fasttext-ranking-v2, fineweb-edu-ranking-v2,
    modernbert-ranking-v2, topic
`text` is read only by the materialize stage, never here.
"""
from __future__ import annotations

import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .io import check, log, stop

V2_FT = "fasttext-ranking-v2"
V2_FW = "fineweb-edu-ranking-v2"
V2_MB = "modernbert-ranking-v2"
NUMERIC_COLS = ["doc_id", "tokens-llama2", V2_FT, V2_FW, V2_MB, "topic"]

# The 24 WebOrganizer categories, as published on the dataset card.
TOPICS_EXPECTED = 24


def _init_worker() -> None:
    pa.set_cpu_count(1)


@dataclass
class Pool:
    """In-memory numeric view of the pool, indexed by doc_id (which equals row position)."""

    n: int
    tok: np.ndarray        # int64, TRAIN length = tokens-llama2 + 1   (the 1.5B convention)
    ft: np.ndarray         # float32 fasttext-ranking-v2   (global tie-aware percentile)
    fw: np.ndarray         # float32 fineweb-edu-ranking-v2
    mb: np.ndarray         # float32 modernbert-ranking-v2
    topic: np.ndarray      # int8 code into `vocab`
    vocab: list[str]

    @property
    def q(self) -> np.ndarray:
        """Consensus quality.  float32 and ddof=0 are load-bearing (dataset card)."""
        return ((self.ft + self.fw + self.mb) / 3.0).astype(np.float32)

    def qv(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.q
        v = (((self.ft - q) ** 2 + (self.fw - q) ** 2 + (self.mb - q) ** 2) / 3.0).astype(
            np.float32
        )
        return q, v

    def score(self, key: str) -> np.ndarray:
        """Named scorer array.  'u' needs lambda and is built in disagreement.py."""
        return {"fasttext": self.ft, "fineweb": self.fw, "modernbert": self.mb, "q": self.q}[key]


def shard_geometry(cfg) -> tuple[int, int, int, int]:
    p = cfg.pool
    return int(p["n_shards"]), int(p["rows_full"]), int(p["rows_last"]), int(p["n_docs"])


def shard_rows(cfg, i: int) -> int:
    n_shards, rows_full, rows_last, _ = shard_geometry(cfg)
    return rows_full if i < n_shards - 1 else rows_last


def shard_offset(cfg, i: int) -> int:
    _, rows_full, _, _ = shard_geometry(cfg)
    return i * rows_full


def doc_id_to_shard(cfg, doc_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(shard_index, row_in_shard) for a doc_id array.  doc_id == shard*rows_full + row."""
    _, rows_full, _, _ = shard_geometry(cfg)
    d = np.asarray(doc_ids, dtype=np.int64)
    return (d // rows_full).astype(np.int32), (d % rows_full).astype(np.int32)


def _read_numeric(args):
    path, expect_rows, lo = args
    t = pq.read_table(path, columns=NUMERIC_COLS, use_threads=False)
    did = t.column("doc_id").to_numpy(zero_copy_only=False)
    # +1 BOS: TRAIN length, the 1.5B budget convention (select_10b.py:92)
    tok = t.column("tokens-llama2").to_numpy(zero_copy_only=False).astype(np.int64) + 1
    ft = t.column(V2_FT).to_numpy(zero_copy_only=False).astype(np.float32)
    fw = t.column(V2_FW).to_numpy(zero_copy_only=False).astype(np.float32)
    mb = t.column(V2_MB).to_numpy(zero_copy_only=False).astype(np.float32)
    dic = t.column("topic").dictionary_encode().combine_chunks()
    local_vocab = dic.dictionary.to_pylist()
    local_codes = dic.indices.to_numpy(zero_copy_only=False).astype(np.int32)
    nulls = {c: t.column(c).null_count for c in NUMERIC_COLS[1:]}
    return (
        path.name, t.num_rows, expect_rows, lo, did, tok, ft, fw, mb,
        local_vocab, local_codes, nulls,
    )


def load(cfg, workers: int | None = None) -> Pool:
    """PASS1: load the numeric columns across all shards, with every 1.5B gate."""
    n_shards, rows_full, rows_last, n = shard_geometry(cfg)
    workers = workers or _default_workers()
    log(f"PASS1: loading numeric cols across {n_shards} shards with {workers} workers...")

    tok = np.empty(n, np.int64)
    ft = np.empty(n, np.float32)
    fw = np.empty(n, np.float32)
    mb = np.empty(n, np.float32)
    topic_local: list = [None] * n_shards
    null_tot: Counter = Counter()

    jobs = []
    for i in range(n_shards):
        p = cfg.pool_shard(i)
        check(p.exists(), f"missing pool shard {p} -- run bin/00_download_pool.py first")
        jobs.append((p, shard_rows(cfg, i), shard_offset(cfg, i)))

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("fork"), initializer=_init_worker
    ) as ex:
        futs = {ex.submit(_read_numeric, j): i for i, j in enumerate(jobs)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            name, nrows, expect, lo, did, tk, a, b, c, lv, lc, nl = fut.result()
            check(nrows == expect, f"{name}: {nrows} rows != expected {expect}")
            hi = lo + nrows
            check(
                bool(np.array_equal(did, np.arange(lo, hi, dtype=did.dtype))),
                f"{name}: doc_id not contiguous at [{lo},{hi})",
            )
            tok[lo:hi] = tk
            ft[lo:hi] = a
            fw[lo:hi] = b
            mb[lo:hi] = c
            topic_local[i] = (lv, lc, lo)
            for k, vv in nl.items():
                null_tot[k] += vv
            done += 1
            if done % 50 == 0:
                log(f"  PASS1 {done}/{n_shards}")

    check(not any(null_tot.values()), f"nulls found in the pool: {dict(null_tot)}")

    vocab = sorted({s for lv, _, _ in topic_local for s in lv})
    if len(vocab) != TOPICS_EXPECTED:
        print("topic vocab:", vocab, flush=True)
        stop(f"distinct topic count {len(vocab)} != {TOPICS_EXPECTED}")
    code_of = {s: k for k, s in enumerate(vocab)}
    topic = np.empty(n, np.int8)
    for lv, lc, lo in topic_local:
        remap = np.array([code_of[s] for s in lv], dtype=np.int8)
        topic[lo : lo + lc.size] = remap[lc]

    for nm, arr in (("fasttext", ft), ("fineweb-edu", fw), ("modernbert", mb)):
        check(
            bool((arr > 0).all() and (arr <= 1.0 + 1e-6).all()),
            f"{nm}-ranking-v2 outside (0,1]: min={arr.min()} max={arr.max()}",
        )

    log(f"PASS1 OK: N={n:,}, topics={len(vocab)}, zero nulls.")
    return Pool(n=n, tok=tok, ft=ft, fw=fw, mb=mb, topic=topic, vocab=vocab)


def _default_workers() -> int:
    import os

    return int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 8))


def validate_files(cfg) -> dict:
    """Stage-00 check: footers only, no data read.  Used by bin/validate.py --stage pool."""
    n_shards, rows_full, rows_last, n_expect = shard_geometry(cfg)
    total = 0
    rows = []
    for i in range(n_shards):
        p = cfg.pool_shard(i)
        check(p.exists(), f"missing pool shard {p}")
        f = pq.ParquetFile(str(p))
        nrows = f.metadata.num_rows
        check(
            nrows == shard_rows(cfg, i),
            f"{p.name}: {nrows} rows != expected {shard_rows(cfg, i)}",
        )
        names = set(f.schema_arrow.names)
        missing = set(NUMERIC_COLS + ["text"]) - names
        check(not missing, f"{p.name}: missing columns {sorted(missing)}")
        total += nrows
        rows.append(nrows)
    check(total == n_expect, f"pool total {total:,} != expected {n_expect:,}")
    return dict(n_shards=n_shards, total_rows=total, rows_per_shard=rows)
