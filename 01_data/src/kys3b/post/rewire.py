"""REWIRE-INSPIRED post-processing: pool BOTH passes -> fastText the REWRITTEN text -> top-B.

This is the one arm where quality selection happens AFTER rewriting.  The 1.5B pipeline
(10_postprocess/02..04_*_rewrite.py) did exactly:

  1. strip prefixes, recount            (post/strip.py, shared)
  2. pool = ALL status==2 pass-1  UNION  ALL status==2 distill
     Both passes are in the pool, one row each, NOT deduplicated by doc_id.  At 1.5B distill
     won 53.1% of the retained tokens (it scores higher under fastText), so dropping it would
     change the corpus, not just cheapen it.  `require_both_passes` enforces this.
  3. fastText-score the REWRITTEN text with the exact 1.5B recipe (fasttext_score.py)
  4. global DESC sort on the RAW score, fill to B (last document whole)
  5. mix with the shared base + shuffle (post/mix.py)

The v2 percentile against the original pool distribution is recorded for readability; the
retained set is identical either way because the map is monotone.  SORTING USES THE RAW SCORE.
"""
from __future__ import annotations

import numpy as np

from ..io import check, log, stop


def require_both_passes(p1_shards: int, distill_shards: int, expected: int) -> None:
    """Hard gate: REWIRE may not be filtered until both passes are complete."""
    check(
        p1_shards == expected and distill_shards == expected,
        f"REWIRE: both passes must be complete before filtering "
        f"(pass1 {p1_shards}/{expected}, distill {distill_shards}/{expected}). "
        "Decision 3: the fastText filter ranks BOTH passes together.",
    )


def build_pool(p1: dict, distill: dict) -> dict:
    """Concatenate the two passes' status==2 rows into one flat candidate pool."""
    s2a = np.flatnonzero(p1["status"] == 2)
    s2b = np.flatnonzero(distill["status"] == 2)
    pool = dict(
        doc_id=np.concatenate([p1["doc_id"][s2a], distill["doc_id"][s2b]]),
        length=np.concatenate([p1["len"][s2a], distill["len"][s2b]]),
        source_prompt=np.concatenate(
            [np.zeros(s2a.size, np.int8), np.ones(s2b.size, np.int8)]  # 0=p1, 1=distill
        ),
        row=np.concatenate([s2a, s2b]),
    )
    log(
        f"  REWIRE pool: {pool['doc_id'].size:,} rewritten docs, "
        f"{int(pool['length'].sum()):,} tok "
        f"(pass1 {s2a.size:,}/{int(p1['len'][s2a].sum()):,}; "
        f"distill {s2b.size:,}/{int(distill['len'][s2b].sum()):,})"
    )
    return pool


def filter_top_b(pool: dict, raw_score: np.ndarray, target: int) -> dict:
    """Global DESC sort on the raw fastText score; fill to `target` rewritten train tokens."""
    n = pool["doc_id"].size
    check(raw_score.size == n, f"score size {raw_score.size} != pool size {n}")
    order = np.argsort(-raw_score.astype(np.float64), kind="stable")
    c = np.cumsum(pool["length"][order])
    if c[-1] < target:
        stop(
            f"REWIRE: rewritten pool supplies only {int(c[-1]):,} tok < target {target:,}. "
            "Both passes must be present; check the funnel report."
        )
    cut = int(np.searchsorted(c, target, side="left"))
    keep = order[: cut + 1]
    kept_mask = np.zeros(n, bool)
    kept_mask[keep] = True
    is_p1 = pool["source_prompt"] == 0

    kept_p1 = kept_mask & is_p1
    kept_d = kept_mask & ~is_p1
    cutoff = float(raw_score[keep[-1]])
    rej = ~kept_mask

    def dist(a):
        if a.size == 0:
            return None
        return dict(
            min=float(a.min()), p10=float(np.percentile(a, 10)),
            median=float(np.median(a)), p90=float(np.percentile(a, 90)),
            max=float(a.max()), n=int(a.size),
        )

    out = dict(
        kept_mask=kept_mask,
        kept_docs=int(kept_mask.sum()),
        kept_tokens=int(pool["length"][kept_mask].sum()),
        target=int(target),
        overshoot=int(pool["length"][kept_mask].sum() - target),
        fasttext_score_cutoff=cutoff,
        kept_from_pass1=dict(docs=int(kept_p1.sum()), tokens=int(pool["length"][kept_p1].sum())),
        kept_from_distill=dict(docs=int(kept_d.sum()), tokens=int(pool["length"][kept_d].sum())),
        pool_docs=int(n),
        pool_tokens=int(pool["length"].sum()),
        kept_score_dist=dist(raw_score[kept_mask]),
        rejected_score_dist=dist(raw_score[rej]),
    )
    kt = out["kept_tokens"]
    out["distill_share_pct"] = 100.0 * out["kept_from_distill"]["tokens"] / kt if kt else 0.0
    out["acceptance_pct_by_token"] = 100.0 * kt / max(1, out["pool_tokens"])
    log(
        f"  REWIRE kept {out['kept_docs']:,} docs / {kt:,} tok at raw cutoff {cutoff:.6g}; "
        f"distill share {out['distill_share_pct']:.1f}% "
        f"(1.5B: 53.1%); acceptance {out['acceptance_pct_by_token']:.1f}% (1.5B: 30.8%)"
    )
    return out


def reference_distribution(cfg, sample_shards: int | None = None) -> tuple[np.ndarray, int]:
    """Sorted raw `fasttext` column over the pool -- the v2 percentile reference.

    `sample_shards` reads only the first N pool shards, for a cheap approximate percentile
    during a report run.  The committed percentile always uses all 200.
    """
    import pyarrow.parquet as pq

    n_shards = int(cfg.pool["n_shards"])
    use = n_shards if sample_shards is None else min(sample_shards, n_shards)
    parts = []
    for i in range(use):
        t = pq.read_table(cfg.pool_shard(i), columns=["fasttext"], use_threads=False)
        parts.append(t.column("fasttext").to_numpy(zero_copy_only=False).astype(np.float32))
    ref = np.sort(np.concatenate(parts))
    return ref, int(ref.size)
