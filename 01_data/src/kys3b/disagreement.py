"""DISAGREEMENT-AWARE: candidate domain U, the Q30/Q90 feasible domain, and the u ranking.

Option C (locked, plan.md section 20 D-2): the per-scorer candidate budget is B_s = 20B.

  for each scorer s in {fasttext, fineweb-edu, modernbert}:
      A_s = fill_to(order_desc(D', r_s), da_per_scorer)      # 20B train tokens
  U     = A_ft | A_fw | A_mb
  tau_q = 30th percentile of q over U
  tau_v = 90th percentile of v over U
  U_tau = {d in U : q >= tau_q and v <= tau_v}
  S_DA  = fill_to(order_desc(U_tau, q + lambda*sqrt(v)), B_s)

lambda, the floor percentile and the cap percentile are the 1.5B values (0.5, 30, 90) and
are never retuned against downstream performance.  Only lambda = 0.5 is built (Decision 12).

The 1.5B rule was "each scorer's top 10% BY TOKENS of the residual pool" (~9.03e9 ~= 0.90*B_s).
At 3B that literal rule yields ~8.53e9 per scorer -> U_tau ~15.8e9 < the 20e9 the arm must
fill, i.e. infeasible; `per_scorer_frac_of_residual` below exists only so the report phase
can print what the old rule WOULD have given, for the record.
"""
from __future__ import annotations

import numpy as np

from .io import check, log, stop
from .select import Block, fill_to_strict, mask_of, order_desc


def build_domain(pool, cfg, residual_idx, q, v, tie, tok) -> dict:
    """Build U and U_tau.  Returns every number the DOMAIN_STATS report needs."""
    per_scorer = cfg.budget("da_per_scorer")
    scorers = (("fasttext", pool.ft), ("fineweb-edu", pool.fw), ("modernbert", pool.mb))

    per = {}
    masks = []
    for name, arr in scorers:
        sel, tot, over = fill_to_strict(
            order_desc(residual_idx, arr, tie), tok, per_scorer, f"DA candidate set ({name})"
        )
        masks.append(mask_of(sel, pool.n))
        per[name] = dict(docs=int(sel.size), tokens=int(tot), overshoot=int(over))
        log(f"  DA domain {name:12s}: {sel.size:,} docs, {tot:,} tok")

    u_mask = masks[0] | masks[1] | masks[2]
    inter3 = masks[0] & masks[1] & masks[2]
    u_idx = np.flatnonzero(u_mask)
    u_tok = int(tok[u_idx].sum())
    log(f"  DA domain U = union: {u_idx.size:,} docs, {u_tok:,} tok")

    qu, vu = q[u_idx], v[u_idx]
    q_floor_pct = float(cfg.budgets["disagreement"]["q_floor_pct"])
    v_cap_pct = float(cfg.budgets["disagreement"]["v_cap_pct"])
    tau_q = float(np.percentile(qu, q_floor_pct))
    tau_v = float(np.percentile(vu, v_cap_pct))

    keep = (qu >= tau_q) & (vu <= tau_v)
    ut_idx = u_idx[keep]
    ut_tok = int(tok[ut_idx].sum())
    log(f"  DA feasible U_tau: {ut_idx.size:,} docs, {ut_tok:,} tok")

    # the floor x cap contingency table (the 1.5B DOMAIN_STATS.md table)
    pass_q, pass_v = qu >= tau_q, vu <= tau_v
    contingency = dict(
        q_pass_v_pass=int(np.count_nonzero(pass_q & pass_v)),
        q_pass_v_fail=int(np.count_nonzero(pass_q & ~pass_v)),
        q_fail_v_pass=int(np.count_nonzero(~pass_q & pass_v)),
        q_fail_v_fail=int(np.count_nonzero(~pass_q & ~pass_v)),
    )

    if ut_tok < cfg.Bs:
        stop(
            f"DISAGREEMENT-AWARE infeasible: U_tau has {ut_tok:,} train tokens < B_s "
            f"({cfg.Bs:,}).  Per-scorer budget is {per_scorer:,} (Option C). "
            "Revisit plan.md section 20 D-2 before changing anything."
        )

    return dict(
        per_scorer_budget=per_scorer,
        per_scorer=per,
        intersection_abc_docs=int(np.count_nonzero(inter3)),
        u_idx=u_idx,
        u_docs=int(u_idx.size),
        u_tokens=u_tok,
        q_floor_pct=q_floor_pct,
        v_cap_pct=v_cap_pct,
        tau_q=tau_q,
        tau_v=tau_v,
        ut_idx=ut_idx,
        ut_docs=int(ut_idx.size),
        ut_tokens=ut_tok,
        ut_pct_of_u_docs=100.0 * ut_idx.size / max(1, u_idx.size),
        ut_pct_of_u_tokens=100.0 * ut_tok / max(1, u_tok),
        contingency=contingency,
    )


def select_disagreement(pool, cfg, residual_idx, q, v, tie, tok) -> Block:
    dom = build_domain(pool, cfg, residual_idx, q, v, tie, tok)
    ut = dom["ut_idx"]
    lam = cfg.lam
    check(lam == 0.5, f"only lambda=0.5 is built (Decision 12); got {lam}")

    u_score = (q[ut] + lam * np.sqrt(v[ut])).astype(np.float32)
    # order_desc over a subset needs the score indexed by doc_id, so scatter it back
    u_full = np.zeros(pool.n, np.float32)
    u_full[ut] = u_score
    order = order_desc(ut, u_full, tie)

    target = cfg.source_budget("disagreement-aware")
    idx, total, over = fill_to_strict(order, tok, target, "disagreement-aware")

    extra = {k: dom[k] for k in dom if not k.endswith("_idx")}
    extra["lambda"] = lam
    extra["pct_of_ut_docs"] = 100.0 * idx.size / max(1, ut.size)
    extra["u_at_cutoff"] = float(u_full[idx[-1]]) if idx.size else None
    extra["selected_q_mean"] = float(q[idx].mean()) if idx.size else None
    extra["selected_v_mean"] = float(v[idx].mean()) if idx.size else None
    return Block(
        "disagreement-aware", idx, total, over, target,
        f"U = union of three per-scorer top-{cfg.budget('da_per_scorer'):,}-token selections over D'; "
        f"tau_q=Q{dom['q_floor_pct']:.0f}(q|U), tau_v=Q{dom['v_cap_pct']:.0f}(v|U); "
        f"ranked by u = q + {lam}*sqrt(v) DESC (Option C)",
        extra=extra,
    )


def u_score_for(q: np.ndarray, v: np.ndarray, lam: float) -> np.ndarray:
    """u = q + lambda*sqrt(v), float32 throughout -- the assembly sort key for DA."""
    return (q + np.float32(lam) * np.sqrt(v)).astype(np.float32)


def qv_from_columns(ft, fw, mb):
    """Consensus quality and disagreement from the three v2 percentile columns.

    float32 and ddof=0 are load-bearing -- this is the code that built the released corpora
    (dataset card, and 10_postprocess/02_assemble_5B.py:87-98).
    """
    ft = np.asarray(ft, dtype=np.float32)
    fw = np.asarray(fw, dtype=np.float32)
    mb = np.asarray(mb, dtype=np.float32)
    q = ((ft + fw + mb) / 3.0).astype(np.float32)
    v = (((ft - q) ** 2 + (fw - q) ** 2 + (mb - q) ** 2) / 3.0).astype(np.float32)
    return q, v
