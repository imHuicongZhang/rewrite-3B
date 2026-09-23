#!/usr/bin/env python
"""Read the calibration pilot and decide whether the yield clears the 10B target.

Reports, per arm: r_census for p1 and distill separately (exact token accounting), the projected
full-arm rewritten-token total, and the headroom against B.  Compares each measured r to the 1.5B
census value so an adverse population shift is visible rather than inferred.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from kys3b.calibration import REF_R, aggregate, pilot_shards, project_yield
from kys3b.config import PASSES, REWRITE_SETTINGS, Config
from kys3b.io import atomic_write_json, log
from kys3b.shards import load_index, shard_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="calibration pilot report")
    ap.add_argument("--settings", nargs="*", default=list(REWRITE_SETTINGS))
    ap.add_argument("--shards", type=int, default=None, metavar="K",
                    help="expected pilot size; default: whatever shards are present")
    ap.add_argument("--from", dest="src", default="pilot",
                    choices=["pilot", "production"], help="which output tree to read")
    args = ap.parse_args(argv)
    cfg = Config.load()

    print(f"{'setting':22s} {'pass':8s} {'shards':>6s} {'src_tok':>15s} {'out_tok':>15s} "
          f"{'r_census':>9s} {'r_1.5B':>8s} {'delta':>8s} {'r_status2':>10s}")
    print("-" * 108)
    results: dict = {}
    for setting in args.settings:
        src_dir = cfg.stage("sources") / setting
        try:
            idx = load_index(src_dir)
        except Exception:
            continue
        n = int(idx["n_shards"])
        want = pilot_shards(n, args.shards) if args.shards else None
        per_pass: dict[str, float] = {}
        detail: dict = {}
        for pass_name in PASSES:
            base = cfg.stage("rewritten") / setting
            out_dir = (base / "_pilot" / pass_name) if args.src == "pilot" else (base / pass_name)
            have = [k for k in range(n) if shard_path(out_dir, k).exists()]
            if want is not None:
                have = [k for k in have if k in set(want)]
            if not have:
                continue
            agg = aggregate(src_dir, out_dir, have)
            ref = REF_R[setting][pass_name]
            per_pass[pass_name] = agg["r_census"]
            detail[pass_name] = dict(shards=have, **agg, reference_r=ref)
            print(
                f"{setting:22s} {pass_name:8s} {len(have):>6} "
                f"{agg['src_train_tokens_all']:>15,} {agg['out_tokens_status2']:>15,} "
                f"{agg['r_census']:>9.4f} {ref:>8.4f} "
                f"{100*(agg['r_census']-ref)/ref:>7.1f}% {agg['r_status2']:>10.4f}"
            )
        if per_pass:
            results[setting] = dict(
                per_pass=detail,
                projection=project_yield(cfg, per_pass, cfg.source_budget(setting)),
            )

    print(f"\nprojected full-arm yield vs the {cfg.B/1e9:.0f}B target:")
    print(f"  {'setting':22s} {'source':>8s} {'proj p1':>10s} {'proj distill':>13s} "
          f"{'proj total':>12s} {'headroom':>10s}")
    blockers = []
    for setting, r in results.items():
        pr = r["projection"]
        if not pr["complete"]:
            print(f"  {setting:22s} (incomplete: needs both p1 and distill)")
            continue
        flag = "" if pr["meets_target"] else "   <-- BELOW TARGET"
        if not pr["meets_target"] or pr["headroom_pct"] < 5.0:
            blockers.append(setting)
        print(
            f"  {setting:22s} {pr['source_budget']/1e9:7.0f}B "
            f"{pr['projected_p1_tokens']/1e9:9.2f}B {pr['projected_distill_tokens']/1e9:12.2f}B "
            f"{pr['projected_total_tokens']/1e9:11.2f}B {pr['headroom_pct']:+9.1f}%{flag}"
        )

    dest = cfg.stage("reports") / f"pilot_{args.src}.json"
    atomic_write_json(dict(source=args.src, target=cfg.B, results=results), dest)
    log(f"written {dest}")

    if blockers:
        print(
            f"\n  GATE: {', '.join(blockers)} project <5% headroom against the "
            f"{cfg.B/1e9:.0f}B target.  Do NOT launch production until this is resolved -- "
            "the arm would finish short and the corpus could not be assembled."
        )
        return 1
    if results:
        print("\n  GATE PASSED: every measured arm projects above the target with margin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
