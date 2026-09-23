"""Collect the light columns of a rewritten pass across all shards, plus the assembly keys.

The assembly sort key is ALWAYS computed from the ORIGINAL document's precomputed columns --
the rewritten text is never scored here (that happens only in the REWIRE arm).  Loading the
keys out of the pool by doc_id guarantees they are byte-identical to what selection used.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from ..io import check
from ..shards import shard_path

LIGHT_COLS = ["doc_id", "rewritten_tokens", "status"]


def collect_pass(pass_dir, n_shards: int, with_style: bool = False) -> dict:
    """Row-parallel arrays over all shards of one pass, plus per-shard row offsets."""
    ids, toks, sts, styles, offs = [], [], [], [], [0]
    for k in range(n_shards):
        p = shard_path(pass_dir, k)
        check(p.exists(), f"missing rewritten shard {p} -- the pass is not complete")
        cols = LIGHT_COLS + (["wrap_style"] if with_style else [])
        t = pq.read_table(p, columns=cols, use_threads=False)
        ids.append(t.column("doc_id").to_numpy(zero_copy_only=False))
        toks.append(t.column("rewritten_tokens").to_numpy(zero_copy_only=False).astype(np.int64))
        sts.append(t.column("status").to_numpy(zero_copy_only=False).astype(np.int8))
        if with_style:
            styles.extend(t.column("wrap_style").to_pylist())
        offs.append(offs[-1] + t.num_rows)
    out = dict(
        doc_id=np.concatenate(ids),
        rewritten_tokens=np.concatenate(toks),
        status=np.concatenate(sts),
        offs=np.array(offs, dtype=np.int64),
        n_shards=n_shards,
    )
    out["len"] = out["rewritten_tokens"] + 1  # TRAIN length: one leading BOS
    if with_style:
        out["wrap_style"] = styles
    return out


def attach_keys(data: dict, pool, cfg, sort_key: str | None, with_topic: bool = False) -> dict:
    """Add the assembly sort key (and topic code) for every row, indexed by doc_id."""
    from .assemble import sort_key_values

    d = data["doc_id"]
    ft, fw, mb = pool.ft[d], pool.fw[d], pool.mb[d]
    data["key"] = (
        sort_key_values(sort_key, ft, fw, mb, cfg.lam)
        if sort_key is not None
        else np.zeros(d.size, np.float32)
    )
    if with_topic:
        data["topic"] = pool.topic[d]
    data["source_tokens"] = pool.tok[d]
    return data


def cross_pass_report(p1: dict, distill: dict, pool) -> dict:
    """The 1.5B cross-pass coverage block (02_assemble_5B.py cross_pass).

    Both passes cover the SAME doc_ids, so pass-1 status is aligned to distill order via
    `paired_wiki_status` (asserting equality, with an explicit doc_id-join fallback).
    """
    from ..io import paired_wiki_status

    w = paired_wiki_status(
        dict(doc_id=p1["doc_id"], status=p1["status"]),
        dict(doc_id=distill["doc_id"], status=distill["status"]),
    )
    d = distill["status"]
    src = pool.tok[distill["doc_id"]]
    s0_both = (w == 0) & (d == 0)
    s1_both = (w == 1) & (d == 1)
    recovered = (w != 2) & (d == 2)
    s2_wiki_not_d = (w == 2) & (d != 2)
    any_s2 = (w == 2) | (d == 2)
    return dict(
        paired_docs=int(d.size),
        status0_both=int(s0_both.sum()),
        status0_both_source_tokens=int(src[s0_both].sum()),
        status1_both=int(s1_both.sum()),
        status1_both_source_tokens=int(src[s1_both].sum()),
        recovered_by_distill=int(recovered.sum()),
        status2_pass1_not_distill=int(s2_wiki_not_d.sum()),
        unique_docs_with_any_status2=int(any_s2.sum()),
        source_coverage_tokens=int(src[any_s2].sum()),
        source_total_tokens=int(src.sum()),
        source_coverage_pct=100.0 * float(src[any_s2].sum()) / max(1, int(src.sum())),
    )
