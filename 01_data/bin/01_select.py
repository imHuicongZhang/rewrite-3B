#!/usr/bin/env python
"""Stage 01 -- the deterministic selection.  Two phases, byte-identical logic.

  --phase report   computes everything, writes ONLY reports/ (no doc_ids, no dataset)
  --phase commit   writes 01_selection/<block>/doc_ids.npy + manifests + reports

The report phase answers, before any GPU time or storage is committed:
  * the exact size of the residual pool D'
  * the Option-C DISAGREEMENT-AWARE domain sizes (U, U_tau, tau_q, tau_v, cutoff)
  * the per-topic DIVERSITY supply against its quota, hence the expected shortfall
  * the pairwise overlap matrix (the 3B analogue of paper Fig. 9 -- it MUST be regenerated
    because REWIRE's independence base rate moves from ~22% to ~47%)
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import numpy as np

from kys3b import manifest
from kys3b.config import SETTINGS, Config
from kys3b.io import atomic_save_npy, atomic_write_json, atomic_write_text, log
from kys3b.pool import load as load_pool
from kys3b.select import jaccard, mask_of, select_all


def _overlap_matrix(sel, n):
    names = list(sel.blocks)
    masks = {k: mask_of(b.idx, n) for k, b in sel.blocks.items()}
    rows = {}
    for a in names:
        sa = int(masks[a].sum())
        rows[a] = {}
        for b in names:
            inter = int(np.count_nonzero(masks[a] & masks[b]))
            rows[a][b] = dict(
                pct_of_row=100.0 * inter / sa if sa else 0.0,
                shared_docs=inter,
                jaccard=jaccard(masks[a], masks[b])[0],
            )
    return rows


def _report_md(cfg, sel, overlap) -> str:
    L = [
        "# rewrite-3B -- selection report",
        "",
        f"- code commit: `{manifest.code_commit()}`  |  generated {manifest.now()}",
        f"- seed {cfg.seed}; budgets are TRAIN tokens = (tokens-llama2 + 1)",
        f"- exclusions: the {sel.val_idx.size:,}-document validation holdout ONLY "
        "(Decision 1: the 5M analysis sample is NOT removed)",
        "",
        "## Pool and residual",
        "",
        "| quantity | docs | train tokens |",
        "|---|---:|---:|",
        f"| pool | {cfg.pool['n_docs']:,} | - |",
        f"| val holdout | {sel.val_idx.size:,} | - |",
        f"| ALIVE | {sel.alive_idx.size:,} | - |",
        f"| shared base | {sel.base.docs:,} | {sel.base.tokens:,} |",
        f"| residual D' | {sel.residual_idx.size:,} | {sel.residual_tokens:,} |",
        "",
        "## Blocks",
        "",
        "| block | target | docs | train tokens | overshoot | % of D' |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    L.append(
        f"| shared-base-10B | {sel.base.target:,} | {sel.base.docs:,} | {sel.base.tokens:,} "
        f"| {sel.base.overshoot:,} | (from ALIVE) |"
    )
    for name in SETTINGS:
        b = sel.blocks[name]
        L.append(
            f"| {name} | {b.target:,} | {b.docs:,} | {b.tokens:,} | {b.overshoot:,} "
            f"| {100.0*b.tokens/sel.residual_tokens:.2f}% |"
        )

    da = sel.blocks["disagreement-aware"].extra
    L += [
        "",
        "## DISAGREEMENT-AWARE feasible domain (Option C)",
        "",
        f"- per-scorer candidate budget: **{da['per_scorer_budget']:,}** train tokens (= B_s)",
        "",
        "| set | docs | train tokens |",
        "|---|---:|---:|",
    ]
    for k, v in da["per_scorer"].items():
        L.append(f"| top-B_s under {k} | {v['docs']:,} | {v['tokens']:,} |")
    L += [
        f"| U (union) | {da['u_docs']:,} | {da['u_tokens']:,} |",
        f"| U_tau (feasible) | {da['ut_docs']:,} | {da['ut_tokens']:,} |",
        "",
        f"- documents in all three per-scorer sets: {da['intersection_abc_docs']:,}",
        f"- `tau_q` = Q{da['q_floor_pct']:.0f}(q over U) = {da['tau_q']:.10f}",
        f"- `tau_v` = Q{da['v_cap_pct']:.0f}(v over U) = {da['tau_v']:.10f}",
        f"- U_tau keeps {da['ut_pct_of_u_docs']:.2f}% of U by document, "
        f"{da['ut_pct_of_u_tokens']:.2f}% by token  (1.5B: 69.65% / 76.08%)",
        f"- lambda = {da['lambda']}; selection is {da['pct_of_ut_docs']:.2f}% of U_tau by document "
        "(1.5B: 53.69%)",
        f"- u at the cutoff: {da['u_at_cutoff']}",
        "",
        "### floor x cap contingency over U",
        "",
        "| | v <= tau_v | v > tau_v |",
        "|---|---:|---:|",
        f"| q >= tau_q | {da['contingency']['q_pass_v_pass']:,} | {da['contingency']['q_pass_v_fail']:,} |",
        f"| q <  tau_q | {da['contingency']['q_fail_v_pass']:,} | {da['contingency']['q_fail_v_fail']:,} |",
    ]

    div = sel.blocks["diversity-oriented"].extra
    L += [
        "",
        "## DIVERSITY-ORIENTED per-topic source selection",
        "",
        f"- q-DESC top-up documents after the per-topic fills: {div['topup_docs']:,}",
        "",
        "| topic | pool docs | pool tokens | quota | selected docs | selected tokens | filled | shortfall |",
        "|---|---:|---:|---:|---:|---:|:--:|---:|",
    ]
    for c in div["categories"]:
        L.append(
            f"| {c['topic']} | {c['pool_docs']:,} | {c['pool_tokens']:,} | {c['quota_tokens']:,.0f} "
            f"| {c['docs']:,} | {c['tokens']:,} | {c['filled']} | {c['shortfall_tokens']:,} |"
        )

    rw = sel.blocks["rewire-inspired"].extra
    L += [
        "",
        "## REWIRE-INSPIRED",
        "",
        f"- kappa = {rw['kappa']}; pre-rewrite pool = {sel.blocks['rewire-inspired'].tokens:,} "
        f"train tokens = **{rw['pct_of_residual']:.2f}% of D'**",
        "- at 1.5B this fraction was 22.15%.  The rise is arithmetic (the pool did not double) "
        "and is accepted (Decision 4); the overlap analysis below must be used in place of the "
        "1.5B App. D.3 figure.",
        "",
        "## Pairwise document overlap (|A cap B| / |A|, percent of the ROW setting)",
        "",
        "| row \\ col | " + " | ".join(SETTINGS) + " |",
        "|---|" + "---:|" * len(SETTINGS),
    ]
    for a in SETTINGS:
        L.append(
            f"| {a} | " + " | ".join(f"{overlap[a][b]['pct_of_row']:.1f}" for b in SETTINGS) + " |"
        )
    L += ["", "## Invariants", "", "- base ∩ strategy = 0 for all six settings: **asserted**",
          "- val ∩ everything = 0: **asserted**",
          "- no duplicate doc_id within a block: **asserted**",
          "- quality-base ⊂ quality-first: **asserted**",
          "- every budget filled, overshoot < one document: **asserted** "
          "(the per-topic Diversity quotas are the one reported, non-fatal exception)", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 01: deterministic selection")
    ap.add_argument("--phase", choices=["report", "commit"], required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = Config.load()
    rep_dir = cfg.stage("reports")
    sel_dir = cfg.stage("selection")

    log(f"phase={args.phase}; pool={cfg.root('pool')}")
    pool = load_pool(cfg, workers=args.workers)
    q, v = pool.qv()
    sel = select_all(pool, cfg, q=q, v=v)
    overlap = _overlap_matrix(sel, pool.n)

    # ---- the machine-readable summary, written in BOTH phases -------------------------
    summary = dict(
        phase=args.phase,
        seed=cfg.seed,
        pool=dict(n_docs=int(pool.n), topics=pool.vocab),
        val_docs=int(sel.val_idx.size),
        alive_docs=int(sel.alive_idx.size),
        residual_docs=int(sel.residual_idx.size),
        residual_tokens=int(sel.residual_tokens),
        base=sel.base.summary(),
        base_doc_ids_sha256=manifest.sha256_ids(sel.base.idx),
        blocks={k: b.summary() for k, b in sel.blocks.items()},
        block_doc_ids_sha256={k: manifest.sha256_ids(b.idx) for k, b in sel.blocks.items()},
        overlap=overlap,
        budgets=cfg.budgets["budgets"],
        training=cfg.budgets["training"],
    )
    rep_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary, rep_dir / f"selection_{args.phase}.json")
    atomic_write_text(_report_md(cfg, sel, overlap), rep_dir / "SELECTION_REPORT.md")
    log(f"reports -> {rep_dir}")

    if args.phase == "report":
        log("report phase complete -- NOTHING was written under 01_selection/")
        return 0

    # ---- commit -----------------------------------------------------------------------
    (sel_dir / "val").mkdir(parents=True, exist_ok=True)
    atomic_save_npy(sel.val_idx.astype(np.int64), sel_dir / "val" / "val_doc_ids.npy")

    def _write_block(name, blk, target, method, extra):
        d = sel_dir / name
        d.mkdir(parents=True, exist_ok=True)
        ids = np.sort(blk.idx.astype(np.int64))
        atomic_save_npy(ids, d / "doc_ids.npy")
        manifest.write(
            d, "01_selection",
            dict(
                block=name, docs=int(ids.size), train_tokens=int(blk.tokens),
                target_tokens=int(target), overshoot_tokens=int(blk.overshoot),
                method=method, seed=cfg.seed,
                doc_ids_sha256=manifest.sha256_ids(ids),
                residual_tokens=int(sel.residual_tokens),
                **extra,
            ),
        )

    _write_block("shared-base-10B", sel.base, sel.base.target, sel.base.method, {})
    for name, blk in sel.blocks.items():
        _write_block(name, blk, blk.target, blk.method, blk.extra)

    manifest.write(
        sel_dir, "01_selection",
        dict(
            seed=cfg.seed,
            pool_docs=int(pool.n),
            val_docs=int(sel.val_idx.size),
            residual_docs=int(sel.residual_idx.size),
            residual_tokens=int(sel.residual_tokens),
            base_doc_ids_sha256=manifest.sha256_ids(sel.base.idx),
            block_doc_ids_sha256=summary["block_doc_ids_sha256"],
            budgets=cfg.budgets["budgets"],
            lambda_da=cfg.lam,
            kappa=cfg.kappa,
            exclusions="validation holdout only (Decision 1)",
        ),
    )
    log(f"commit complete -> {sel_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
