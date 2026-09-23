"""Exact compression-ratio accounting for the pre-production calibration gate.

The denominator is summed from the `tokens_llama2` column carried in the materialized source
shards -- never estimated from a row fraction.  Source shards are deterministic slices of a
doc_id-sorted selection, so the first N shards are NOT guaranteed to share the full selection's
mean document length, and `source_budget * sampled_rows / total_rows` would bias r by exactly
that difference.

Two ratios, because they answer different questions and only one is comparable to 1.5B:

  r_census  = sum(rewritten_tokens | status==2) / sum(source tokens_llama2 + 1 | ALL rows)
              The 1.5B census definition -- output totals over the whole per-arm source budget,
              including source documents that were dropped or truncated.  Compare this to
              `REF_R`, and use it to project a full-arm yield against the source budget.

  r_status2 = sum(rewritten_tokens | status==2) / sum(source tokens_llama2 + 1 | status==2 rows)
              Pure per-document compression over exactly the documents that produced output.
              Always >= r_census; the gap is the status-0/1 share.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from .io import check
from .shards import shard_path

# 1.5B census (rewrite-vllm docs/DESIGN_DELTA.md section 5) -- exact, per arm per pass.
# Same definition as r_census.
REF_R = {
    "quality-first": dict(p1=0.3399, distill=0.2581),
    "diversity-oriented": dict(p1=0.3812, distill=0.2845),
    "disagreement-aware": dict(p1=0.3649, distill=0.2750),
    "wrap-inspired": dict(p1=0.4313, distill=0.3657),
    "rewire-inspired": dict(p1=0.4628, distill=0.3660),
}

AGG_KEYS = (
    "rows",
    "src_train_tokens_all",
    "src_train_tokens_status2",
    "out_tokens_status2",
    "docs_status2",
)


def shard_ratio(src_dir, out_dir, k: int) -> dict:
    """Exact token accounting for one shard.  No estimation anywhere."""
    src = pq.read_table(
        shard_path(src_dir, k), columns=["doc_id", "tokens_llama2"], use_threads=False
    )
    out = pq.read_table(
        shard_path(out_dir, k), columns=["doc_id", "status", "rewritten_tokens"], use_threads=False
    )
    s_ids = src.column("doc_id").to_numpy(zero_copy_only=False)
    o_ids = out.column("doc_id").to_numpy(zero_copy_only=False)
    check(
        bool(np.array_equal(s_ids, o_ids)),
        f"shard {k}: rewritten rows are not aligned to the source rows",
    )
    src_train = src.column("tokens_llama2").to_numpy(zero_copy_only=False).astype(np.int64) + 1
    st = out.column("status").to_numpy(zero_copy_only=False)
    rt = out.column("rewritten_tokens").to_numpy(zero_copy_only=False).astype(np.int64)
    s2 = st == 2
    return dict(
        rows=int(s_ids.size),
        src_train_tokens_all=int(src_train.sum()),
        src_train_tokens_status2=int(src_train[s2].sum()),
        out_tokens_status2=int(rt[s2].sum()),
        docs_status2=int(s2.sum()),
    )


def aggregate(src_dir, out_dir, shards) -> dict:
    agg = {k: 0 for k in AGG_KEYS}
    for k in shards:
        r = shard_ratio(src_dir, out_dir, k)
        for key in AGG_KEYS:
            agg[key] += r[key]
    agg["r_census"] = agg["out_tokens_status2"] / max(1, agg["src_train_tokens_all"])
    agg["r_status2"] = agg["out_tokens_status2"] / max(1, agg["src_train_tokens_status2"])
    return agg
