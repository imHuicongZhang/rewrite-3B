"""STEP 2 -- assemble ~B rewritten training tokens per setting.

The 1.5B rule (10_postprocess/02_assemble_5B*.py), preserved exactly:

  take ALL status==2 pass-1 outputs
  if that is still < B:  top up from status==2 distill until sum(rewritten_tokens+1) >= B
  if wiki+distill < B:   report the shortfall and STOP (never pad)

The 1.5B code also contains the `pass-1 alone already fills B` branch (02_assemble_5B.py:269),
which fills from pass 1 in quality order and takes no distill.  It never fired at 1.5B and is
not projected to fire at 3B, but it is preserved so behaviour is identical in either case.

The same doc_id rewritten by two prompts is kept as TWO training examples -- distill is NOT
deduplicated against pass 1.  This is intentional (1.5B DATASETS_SUMMARY section 3).

Per-arm distill top-up ORDER -- these differ, and the differences are load-bearing:
  quality-first        `fasttext-ranking-v2` DESC          (of the ORIGINAL document)
  disagreement-aware   `u = q + 0.5*sqrt(v)` DESC, float32, ddof=0
  diversity-oriented   per-topic quota; `fasttext-ranking-v2` DESC WITHIN topic; policy A:
                       no cross-topic backfill, no padding
  wrap-inspired        SEEDED RANDOM `default_rng(42)` -- deliberately NO quality signal, so the
                       arm stays a "does source quality matter?" baseline
"""
from __future__ import annotations

import numpy as np

from ..disagreement import qv_from_columns
from ..io import log

BOS = 1


def sort_key_values(key: str, ft, fw, mb, lam: float = 0.5):
    """The per-arm assembly sort key, always computed on the ORIGINAL document's columns.

    The rewritten text is never scored here -- that only happens in the REWIRE arm.
    """
    if key == "fasttext":
        return np.asarray(ft, dtype=np.float32)
    if key == "u":
        q, v = qv_from_columns(ft, fw, mb)
        return (q + np.float32(lam) * np.sqrt(v)).astype(np.float32)
    if key is None:
        return None
    raise KeyError(f"unknown assembly sort key {key!r}")


def fill_desc(idx: np.ndarray, key: np.ndarray, length: np.ndarray, target: float):
    """Order `idx` by `key` DESC (stable), cumulative-fill `length` until >= target.

    Verbatim from 02_assemble_5B.py:152-164 -- a plain stable sort, NO seeded tie-break
    (only the quality ordering matters at assembly time), last document kept whole.
    """
    if idx.size == 0:
        return idx, 0, False
    order = idx[np.argsort(-key[idx], kind="stable")]
    c = np.cumsum(length[order])
    if c[-1] < target:
        return order, int(c[-1]), False
    cut = int(np.searchsorted(c, target, side="left"))
    return order[: cut + 1], int(c[cut]), True


def fill_order(idx: np.ndarray, order: np.ndarray, length: np.ndarray, target: float):
    """Cumulative-fill along an explicitly given order (used by the wrap random draw)."""
    if order.size == 0:
        return order, 0, False
    c = np.cumsum(length[order])
    if c[-1] < target:
        return order, int(c[-1]), False
    cut = int(np.searchsorted(c, target, side="left"))
    return order[: cut + 1], int(c[cut]), True


def assemble_flat(p1: dict, distill: dict, target: int, mode: str, seed: int = 42) -> dict:
    """Assemble one non-topic arm.

    `p1` / `distill` are dicts of row-parallel arrays with at least
    doc_id, status, len (= rewritten_tokens + 1) and, for keyed modes, `key`.
    Returns keep masks plus the full accounting.
    """
    s2_p1 = np.flatnonzero(p1["status"] == 2)
    p1_tokens = int(p1["len"][s2_p1].sum())
    gap = target - p1_tokens
    keep_p1 = np.zeros(p1["doc_id"].size, bool)
    keep_d = np.zeros(distill["doc_id"].size, bool)
    log(f"  pass1 status2: {s2_p1.size:,} docs, {p1_tokens:,} tok; gap to target = {gap:,}")

    if p1_tokens >= target:
        # the 1.5B `wiki alone fills B` branch -- never fired at 1.5B
        if mode == "pass1_then_random":
            order = s2_p1[np.random.default_rng(seed).permutation(s2_p1.size)]
            sel, tok_p1, filled = fill_order(s2_p1, order, p1["len"], target)
        else:
            sel, tok_p1, filled = fill_desc(s2_p1, p1["key"], p1["len"], target)
        keep_p1[sel] = True
        log(f"  pass1 alone fills the budget -> {sel.size:,} docs, {tok_p1:,} tok (no distill)")
        return dict(
            keep_p1=keep_p1, keep_distill=keep_d,
            p1_docs=int(sel.size), p1_tokens=int(tok_p1),
            distill_docs=0, distill_tokens=0,
            total_tokens=int(tok_p1), target=int(target),
            shortfall=0, overshoot=int(tok_p1 - target), used_distill=False,
        )

    keep_p1[s2_p1] = True
    s2_d = np.flatnonzero(distill["status"] == 2)
    if mode == "pass1_then_random":
        # deliberately NO quality signal, independent of the Step-3 shuffle RNG
        order = s2_d[np.random.default_rng(seed).permutation(s2_d.size)]
        sel_d, tok_d, filled = fill_order(s2_d, order, distill["len"], gap)
    elif mode == "pass1_then_key":
        sel_d, tok_d, filled = fill_desc(s2_d, distill["key"], distill["len"], gap)
    else:
        raise KeyError(mode)
    keep_d[sel_d] = True
    total = p1_tokens + int(tok_d)
    shortfall = max(0, target - total)
    if shortfall:
        log(
            f"  SHORTFALL: pass1+distill = {total:,} < target {target:,} "
            f"(short by {shortfall:,}) -- no padding is ever applied"
        )
    return dict(
        keep_p1=keep_p1, keep_distill=keep_d,
        p1_docs=int(s2_p1.size), p1_tokens=int(p1_tokens),
        distill_docs=int(sel_d.size), distill_tokens=int(tok_d),
        total_tokens=int(total), target=int(target),
        shortfall=int(shortfall), overshoot=int(max(0, total - target)),
        used_distill=True,
    )


def assemble_per_topic(p1: dict, distill: dict, target: int, topic_proportions: dict) -> dict:
    """DIVERSITY-ORIENTED assembly, policy A (1.5B 02_assemble_5B_diversity.py).

    Topic proportions come from the SELECTED SOURCE SET (tokens-llama2 + 1), not from the pool.
    Each topic is filled ONLY from its own documents: all status==2 pass-1 first, then a distill
    top-up ordered by `fasttext-ranking-v2` DESC WITHIN topic.  No cross-topic backfill, no
    padding.  A topic that exhausts its own supply is reported as a shortfall.
    """
    keep_p1 = np.zeros(p1["doc_id"].size, bool)
    keep_d = np.zeros(distill["doc_id"].size, bool)
    rows = []
    total = 0
    for topic, prop in sorted(topic_proportions.items()):
        quota = target * float(prop)
        tp1 = np.flatnonzero((p1["status"] == 2) & (p1["topic"] == topic))
        td = np.flatnonzero((distill["status"] == 2) & (distill["topic"] == topic))
        p1_tok = int(p1["len"][tp1].sum())
        if p1_tok >= quota:
            sel, tok_here, _ = fill_desc(tp1, p1["key"], p1["len"], quota)
            keep_p1[sel] = True
            d_docs = d_tok = 0
            got = int(tok_here)
            p1_docs, p1_used = int(sel.size), int(tok_here)
        else:
            keep_p1[tp1] = True
            sel_d, d_tok, _ = fill_desc(td, distill["key"], distill["len"], quota - p1_tok)
            keep_d[sel_d] = True
            d_docs = int(sel_d.size)
            got = p1_tok + int(d_tok)
            p1_docs, p1_used = int(tp1.size), int(p1_tok)
        short = max(0, int(round(quota)) - got)
        total += got
        rows.append(
            dict(
                topic=topic, quota_tokens=float(quota),
                p1_docs=p1_docs, p1_tokens=p1_used,
                distill_docs=int(d_docs), distill_tokens=int(d_tok),
                tokens=int(got), shortfall_tokens=int(short),
            )
        )
    shortfall = max(0, target - total)
    short_topics = [r for r in rows if r["shortfall_tokens"] > 0]
    if shortfall:
        log(
            f"  DIVERSITY shortfall {shortfall:,} tok across {len(short_topics)} topics "
            "(policy A: topic balance is preserved; NO cross-topic backfill, NO padding)"
        )
        for r in sorted(short_topics, key=lambda r: -r["shortfall_tokens"])[:8]:
            log(f"    {str(r['topic']):24s} short {r['shortfall_tokens']:,}")
    return dict(
        keep_p1=keep_p1, keep_distill=keep_d,
        p1_docs=int(keep_p1.sum()), p1_tokens=int(p1["len"][keep_p1].sum()),
        distill_docs=int(keep_d.sum()), distill_tokens=int(distill["len"][keep_d].sum()),
        total_tokens=int(total), target=int(target),
        shortfall=int(shortfall), overshoot=int(max(0, total - target)),
        used_distill=bool(keep_d.any()), per_topic=rows,
        shortfall_topics=[r["topic"] for r in short_topics],
    )
