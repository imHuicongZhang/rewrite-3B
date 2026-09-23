#!/usr/bin/env python
"""Measure the realized compression ratio r per (setting, pass) from completed shards.

plan.md section 18.2 projects the 3B yields from the 1.5B census ratios, and flags the
`r`-transfer risk: QUALITY-FIRST has only +19.6% headroom, so a ~16% adverse shift would put
it under budget.  This script re-measures r on whatever shards exist, so the projection can be
checked on ~1% of the work before the fleet is scaled up.

r = sum(rewritten_tokens over status==2) / sum(source tokens-llama2+1 over the same shards)
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pyarrow.parquet as pq

from kys3b.config import PASSES, REWRITE_SETTINGS, Config
from kys3b.shards import load_index, shard_path

# 1.5B census (rewrite-vllm docs/DESIGN_DELTA.md section 5) -- exact, per arm per pass
REF_R = {
    "quality-first": dict(p1=0.3399, distill=0.2581),
    "diversity-oriented": dict(p1=0.3812, distill=0.2845),
    "disagreement-aware": dict(p1=0.3649, distill=0.2750),
    "wrap-inspired": dict(p1=0.4313, distill=0.3657),
    "rewire-inspired": dict(p1=0.4628, distill=0.3660),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="measure realized compression ratios")
    ap.add_argument("--settings", nargs="*", default=list(REWRITE_SETTINGS))
    ap.add_argument("--max-shards", type=int, default=50)
    args = ap.parse_args(argv)
    cfg = Config.load()

    print(
        f"{'setting':22s} {'pass':8s} {'shards':>7s} {'src_tok':>15s} {'out_tok':>15s} "
        f"{'r':>7s} {'r_1.5B':>7s} {'delta':>8s}"
    )
    proj = {}
    for setting in args.settings:
        src_dir = cfg.stage("sources") / setting
        try:
            idx = load_index(src_dir)
        except Exception:
            continue
        budget = cfg.source_budget(setting)
        proj[setting] = dict(source_budget=budget, r={})
        for pass_name in PASSES:
            d = cfg.stage("rewritten") / setting / pass_name
            have = [k for k in range(idx["n_shards"]) if shard_path(d, k).exists()][
                : args.max_shards
            ]
            if not have:
                continue
            out_tok = 0
            for k in have:
                t = pq.read_table(
                    shard_path(d, k), columns=["status", "rewritten_tokens"], use_threads=False
                )
                st = t.column("status").to_numpy(zero_copy_only=False)
                rt = t.column("rewritten_tokens").to_numpy(zero_copy_only=False).astype(np.int64)
                out_tok += int(rt[st == 2].sum())
            # Source tokens for the SAME shards.  Shards are equal-sized slices of a
            # doc_id-sorted selection, so the document share is an unbiased estimator of the
            # token share; this avoids re-reading the pool just to calibrate.
            frac = sum(idx["shards"][k]["rows"] for k in have) / idx["total_docs"]
            src_tok = int(budget * frac)
            r = out_tok / src_tok if src_tok else 0.0
            ref = REF_R[setting][pass_name]
            print(
                f"{setting:22s} {pass_name:8s} {len(have):>7} {src_tok:>15,} {out_tok:>15,} "
                f"{r:>7.4f} {ref:>7.4f} {100*(r-ref)/ref:>7.1f}%"
            )
            proj[setting]["r"][pass_name] = r

    print("\nprojected full-arm yield (measured r x source budget):")
    for s, p in proj.items():
        if len(p["r"]) == 2:
            tot = sum(p["r"].values()) * p["source_budget"]
            head = 100.0 * (tot / cfg.B - 1.0)
            flag = "" if head > 0 else "   <-- BELOW BUDGET"
            print(
                f"  {s:22s} {tot/1e9:7.2f} B vs target {cfg.B/1e9:.0f} B  "
                f"headroom {head:+6.1f}%{flag}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
