"""Selection primitives and the six strategy selectors.

Every primitive is ported verbatim in behaviour from 1.5B `04_select/select_10b.py:145-178`
and its descendants.  The invariants this module enforces:

  * budgets are TRAIN tokens, (tokens-llama2 + 1)
  * `fill_to` keeps the last document whole; an under-fill is a hard stop(), never a
    silent truncation
  * `order_desc` breaks ties with the seed-42 child-1 permutation.  This is load-bearing:
    the DCLM fastText percentile has an 8,655,073-document tie at its floor
  * the shared base is removed before ANY strategy-specific selection, and
    `|BASE cap S| == 0` is asserted for every S
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .io import check, log, stop
from .pool import Pool

# ---- the documented seed-42 child map (identical to select_10b.py:70-77) ----------------
CHILD_VAL = 0      # validation holdout
CHILD_TIE = 1      # global tie-break priority for every DESC ranking
CHILD_WRAP = 2     # WRAP uniform draw
CHILD_REWIRE = 3   # REWIRE uniform draw
N_CHILDREN = 8


def rng_children(seed: int) -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(N_CHILDREN)]


# --------------------------------------------------------------------------- primitives


def order_desc(idxs: np.ndarray, score: np.ndarray, tie: np.ndarray) -> np.ndarray:
    """`idxs` sorted by `score` DESC, ties broken by `tie` ASC (deterministic).

    Verbatim from select_10b.py:146-148.  float64 upcast of the score is part of it.
    """
    return idxs[np.lexsort((tie[idxs], -score[idxs].astype(np.float64)))]


def fill_to(order: np.ndarray, tok: np.ndarray, target: float) -> tuple[np.ndarray, int, int, bool]:
    """Accumulate `tok` along `order` until the cumsum first reaches `target`.

    The last document is kept WHOLE, so a filled block overshoots by at most one document
    minus one token.  Returns (selected, total_tokens, overshoot, filled).
    Verbatim from select_10b.py:150-159.  `.copy()` is applied by callers that keep the
    result, because the raw return is a view that would pin the whole order array alive.
    """
    if order.size == 0:
        return order, 0, int(-target), False
    c = np.cumsum(tok[order])
    if c[-1] < target:
        return order, int(c[-1]), int(c[-1] - target), False
    i = int(np.searchsorted(c, target, side="left"))
    return order[: i + 1], int(c[i]), int(c[i] - target), True


def fill_to_strict(order, tok, target, what: str):
    """`fill_to` that stops the run on an under-fill.  Used for every budget except the
    per-topic Diversity quotas, where a shortfall is the reported policy-A outcome."""
    sel, total, over, filled = fill_to(order, tok, target)
    if not filled:
        stop(
            f"{what}: could not reach {int(target):,} train tokens "
            f"(pool supplies only {total:,})"
        )
    return sel.copy(), total, over


def mask_of(idx: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    m[idx] = True
    return m


def jaccard(a_mask: np.ndarray, b_mask: np.ndarray) -> tuple[float, int, int]:
    inter = int(np.count_nonzero(a_mask & b_mask))
    union = int(np.count_nonzero(a_mask | b_mask))
    return (inter / union if union else 0.0), inter, union


# --------------------------------------------------------------------------- result type


@dataclass
class Block:
    name: str
    idx: np.ndarray
    tokens: int
    overshoot: int
    target: int
    method: str
    extra: dict = field(default_factory=dict)

    @property
    def docs(self) -> int:
        return int(self.idx.size)

    def summary(self) -> dict:
        return dict(
            block=self.name,
            docs=self.docs,
            train_tokens=self.tokens,
            target_tokens=self.target,
            overshoot_tokens=self.overshoot,
            method=self.method,
            **self.extra,
        )


# --------------------------------------------------------------------------- the selection


@dataclass
class Selection:
    """The full deterministic selection: base, residual pool, and the six blocks."""

    val_idx: np.ndarray
    alive_idx: np.ndarray
    base: Block
    residual_idx: np.ndarray
    residual_tokens: int
    blocks: dict[str, Block]
    tie: np.ndarray


def select_all(pool: Pool, cfg, q=None, v=None) -> Selection:
    """Run the whole selection.  Pure given (pool, cfg) -- no IO, no randomness beyond seed."""
    n = pool.n
    seed = cfg.seed
    ch = rng_children(seed)
    tok = pool.tok
    if q is None or v is None:
        q, v = pool.qv()

    tie = ch[CHILD_TIE].permutation(n).astype(np.int64)

    # ---- STEP 0: validation holdout.  Decision 1: this is the ONLY exclusion. -----------
    val_size = int(cfg.pool["val_size"])
    val_idx = np.sort(ch[CHILD_VAL].choice(n, size=val_size, replace=False)).astype(np.int64)
    alive_mask = np.ones(n, bool)
    alive_mask[val_idx] = False
    alive_idx = np.flatnonzero(alive_mask)
    log(f"STEP0 val: {val_idx.size:,} docs; alive={alive_idx.size:,}")

    # ---- shared fixed base: top-10B by fastText percentile over ALIVE -------------------
    base_target = cfg.budget("shared_base")
    avail_order = order_desc(alive_idx, pool.ft, tie)
    base_idx, base_tok, base_over = fill_to_strict(avail_order, tok, base_target, "shared base")
    base = Block(
        name="shared-base-10B",
        idx=base_idx,
        tokens=base_tok,
        overshoot=base_over,
        target=base_target,
        method="fasttext-ranking-v2 DESC over (all minus val)",
    )
    n_base = base_idx.size
    log(f"shared base: {n_base:,} docs, {base_tok:,} tok (overshoot {base_over:,})")

    # ---- residual candidate pool D' = ALIVE \ BASE --------------------------------------
    base_mask = mask_of(base_idx, n)
    residual_mask = alive_mask & ~base_mask
    residual_idx = np.flatnonzero(residual_mask)
    residual_tok = int(tok[residual_idx].sum())
    log(f"RESIDUAL D': {residual_idx.size:,} docs, {residual_tok:,} tok")

    blocks: dict[str, Block] = {}

    # ---- QUALITY-BASE: next 10B along the SAME fastText order ---------------------------
    prefix_order = avail_order[n_base:]
    qb_target = cfg.budget("quality_base")
    qb_idx, qb_tok, qb_over = fill_to_strict(prefix_order, tok, qb_target, "quality-base")
    blocks["quality-base"] = Block(
        "quality-base", qb_idx, qb_tok, qb_over, qb_target,
        "fasttext-ranking-v2 DESC prefix continuation after the base (raw, never rewritten)",
    )

    # ---- QUALITY-FIRST: next 20B along the SAME order (contains quality-base) -----------
    qf_target = cfg.source_budget("quality-first")
    qf_idx, qf_tok, qf_over = fill_to_strict(prefix_order, tok, qf_target, "quality-first")
    blocks["quality-first"] = Block(
        "quality-first", qf_idx, qf_tok, qf_over, qf_target,
        "fasttext-ranking-v2 DESC prefix continuation after the base",
    )

    # ---- WRAP-INSPIRED: uniform from D' -------------------------------------------------
    w_target = cfg.source_budget("wrap-inspired")
    w_order = residual_idx[ch[CHILD_WRAP].permutation(residual_idx.size)]
    w_idx, w_tok, w_over = fill_to_strict(w_order, tok, w_target, "wrap-inspired")
    blocks["wrap-inspired"] = Block(
        "wrap-inspired", w_idx, w_tok, w_over, w_target,
        f"uniform over D' (SeedSequence({seed}) child {CHILD_WRAP})",
    )
    del w_order

    # ---- REWIRE-INSPIRED: uniform from D', kappa * B_s ---------------------------------
    r_target = cfg.source_budget("rewire-inspired")
    check(
        r_target == cfg.kappa * cfg.Bs,
        f"REWIRE budget {r_target:,} != kappa({cfg.kappa}) * B_s({cfg.Bs:,})",
    )
    if residual_tok < r_target:
        stop(
            f"rewire-inspired: D' has {residual_tok:,} tok < {r_target:,} required "
            f"(kappa={cfg.kappa} * B_s={cfg.Bs:,})"
        )
    r_order = residual_idx[ch[CHILD_REWIRE].permutation(residual_idx.size)]
    r_idx, r_tok, r_over = fill_to_strict(r_order, tok, r_target, "rewire-inspired")
    blocks["rewire-inspired"] = Block(
        "rewire-inspired", r_idx, r_tok, r_over, r_target,
        f"uniform over D' (SeedSequence({seed}) child {CHILD_REWIRE}), kappa={cfg.kappa} * B_s",
        extra=dict(kappa=cfg.kappa, pct_of_residual=100.0 * r_tok / residual_tok),
    )
    del r_order

    # ---- DIVERSITY-ORIENTED: topic-stratified, consensus q WITHIN topic -----------------
    blocks["diversity-oriented"] = _select_diversity(
        pool, cfg, residual_idx, residual_tok, q, tie, tok
    )

    # ---- DISAGREEMENT-AWARE -------------------------------------------------------------
    from .disagreement import select_disagreement

    blocks["disagreement-aware"] = select_disagreement(
        pool, cfg, residual_idx, q, v, tie, tok
    )

    # ---- hard invariant: no base document re-enters ANY strategy pool -------------------
    for name, blk in blocks.items():
        overlap = int(np.count_nonzero(base_mask[blk.idx]))
        check(overlap == 0, f"{name}: {overlap:,} documents overlap the shared base")
        val_overlap = int(np.count_nonzero(~alive_mask[blk.idx]))
        check(val_overlap == 0, f"{name}: {val_overlap:,} documents overlap the val holdout")
        check(
            int(np.unique(blk.idx).size) == blk.idx.size,
            f"{name}: duplicate doc_id within the block",
        )
    check(
        bool(np.isin(blocks["quality-base"].idx, blocks["quality-first"].idx).all()),
        "quality-base must be contained in quality-first (both prefixes of the same order)",
    )

    return Selection(
        val_idx=val_idx,
        alive_idx=alive_idx,
        base=base,
        residual_idx=residual_idx,
        residual_tokens=residual_tok,
        blocks=blocks,
        tie=tie,
    )


def _select_diversity(pool, cfg, residual_idx, residual_tok, q, tie, tok) -> Block:
    """Topic-stratified selection, ranked by CONSENSUS q within topic (paper Eq. 6).

    Ported from select_10b.py:254-283, including the q-DESC top-up from D' when the union
    of the per-topic fills lands under the budget.
    """
    target = cfg.source_budget("diversity-oriented")
    topic_n = int(cfg.pool["topic_n"])
    rem_topic = pool.topic[residual_idx]
    sel_parts, cats = [], []
    picked = np.zeros(pool.n, bool)

    for c in range(topic_n):
        cat_idx = residual_idx[rem_topic == c]
        cat_tok = int(tok[cat_idx].sum())
        quota = target * (cat_tok / residual_tok)
        order_c = order_desc(cat_idx, q, tie)
        sel_c, tok_c, over_c, filled_c = fill_to(order_c, tok, quota)
        sel_c = sel_c.copy()
        sel_parts.append(sel_c)
        picked[sel_c] = True
        cats.append(
            dict(
                topic=pool.vocab[c],
                pool_docs=int(cat_idx.size),
                pool_tokens=cat_tok,
                quota_tokens=float(quota),
                docs=int(sel_c.size),
                tokens=int(tok_c),
                filled=bool(filled_c),
                shortfall_tokens=max(0, int(round(quota)) - int(tok_c)),
            )
        )

    idx = np.concatenate(sel_parts)
    total = int(tok[idx].sum())
    topup_docs = 0
    if total < target:
        pool_left = residual_idx[~picked[residual_idx]]
        add_order = order_desc(pool_left, q, tie)
        add_idx, add_tok, _, _ = fill_to(add_order, tok, target - total)
        idx = np.concatenate([idx, add_idx.copy()])
        total += int(add_tok)
        topup_docs = int(add_idx.size)
    return Block(
        "diversity-oriented", idx, total, total - target, target,
        "topic-stratified over D', quota proportional to topic token share, consensus q DESC "
        "within topic, then q-DESC top-up (1.5B select_10b.py behaviour)",
        extra=dict(categories=cats, topup_docs=topup_docs),
    )
