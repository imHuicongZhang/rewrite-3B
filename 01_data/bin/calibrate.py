#!/usr/bin/env python
"""Measure the realized compression ratio r per (setting, pass) from completed shards.

plan.md section 18.2 projects the 3B yields from the 1.5B census ratios and flags the
`r`-transfer risk: QUALITY-FIRST has only +19.6% headroom, so a ~16% adverse shift would put it
under budget.  This re-measures r on whatever shards exist, so the projection can be checked on
~1% of the work before the fleet is scaled up.

**Exact token accounting.**  The denominator is summed from the `tokens_llama2` column carried in
the materialized source shards -- never estimated as `source_budget * sampled_rows / total_rows`.
Source shards are deterministic slices of a doc_id-sorted selection, so the first N shards are
NOT guaranteed to share the full selection's mean document length, and a row-fraction proxy
would bias r by exactly that difference.

Two ratios are reported, because they answer different questions and only one is comparable to
the 1.5B reference:

  r_census  = sum(rewritten_tokens | status==2) / sum(source tokens_llama2 + 1 | ALL rows)
              The 1.5B census definition (output totals over the whole per-arm source budget,
              including the source documents that were dropped or truncated).  This is the
              number to compare against REF_R, and the correct multiplier for projecting a
              full-arm yield against the source budget.

  r_status2 = sum(rewritten_tokens | status==2) / sum(source tokens_llama2 + 1 | status==2 rows)
              Pure per-document compression over exactly the documents that produced output.
              Always >= r_census; the gap is the status-0/1 share.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)
from kys3b.calibration import REF_R, aggregate
from kys3b.config import PASSES, REWRITE_SETTINGS, Config
from kys3b.shards import load_index, shard_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="measure realized compression ratios (exact tokens)")
    ap.add_argument("--settings", nargs="*", default=list(REWRITE_SETTINGS))
    ap.add_argument("--max-shards", type=int, default=50)
    args = ap.parse_args(argv)
    cfg = Config.load()

    print(
        f"{'setting':22s} {'pass':8s} {'shards':>6s} {'src_tok(all)':>14s} {'out_tok':>14s} "
        f"{'r_census':>9s} {'r_1.5B':>8s} {'delta':>8s} {'r_status2':>10s}"
    )
    proj: dict = {}
    for setting in args.settings:
        src_dir = cfg.stage("sources") / setting
        try:
            idx = load_index(src_dir)
        except Exception:
            continue
        budget = cfg.source_budget(setting)
        proj[setting] = dict(source_budget=budget, r={})
        for pass_name in PASSES:
            out_dir = cfg.stage("rewritten") / setting / pass_name
            have = [k for k in range(idx["n_shards"]) if shard_path(out_dir, k).exists()][
                : args.max_shards
            ]
            if not have:
                continue
            agg = aggregate(src_dir, out_dir, have)
            r_census, r_s2 = agg["r_census"], agg["r_status2"]
            ref = REF_R[setting][pass_name]
            print(
                f"{setting:22s} {pass_name:8s} {len(have):>6} {agg['src_train_tokens_all']:>14,} "
                f"{agg['out_tokens_status2']:>14,} {r_census:>9.4f} {ref:>8.4f} "
                f"{100*(r_census-ref)/ref:>7.1f}% {r_s2:>10.4f}"
            )
            proj[setting]["r"][pass_name] = r_census

    print("\nprojected full-arm yield (measured r_census x source budget):")
    tight = []
    for s, p in proj.items():
        if len(p["r"]) == 2:
            tot = sum(p["r"].values()) * p["source_budget"]
            head = 100.0 * (tot / cfg.B - 1.0)
            flag = "" if head > 0 else "   <-- BELOW BUDGET"
            if head <= 10.0:
                tight.append(s)
            print(
                f"  {s:22s} {tot/1e9:7.2f} B vs target {cfg.B/1e9:.0f} B  "
                f"headroom {head:+6.1f}%{flag}"
            )
    if tight:
        print(
            f"\n  WARNING: {', '.join(tight)} have <=10% headroom. plan.md section 18.2 projects "
            "quality-first at +19.6%, the tightest arm; re-check before scaling the fleet."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
