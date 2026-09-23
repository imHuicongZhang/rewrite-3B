#!/usr/bin/env python
"""Read the calibration pilot and decide whether the yield clears the B target.

**This gate fails closed.**  A projection is produced only when, for every requested setting and
BOTH passes, every one of the deterministic `pilot_shards(n, K)` ids is present, full-row, and
doc_id-aligned with its source shard.  Anything less -- a missing shard, a partial shard, a
setting with only one pass, no outputs at all -- prints exactly what is wrong, refuses to print
GATE PASSED, and exits non-zero.  Calibrating from whatever happens to exist is precisely the way
a gate lets thousands of GPU-hours through on a biased subset.

Reports per arm: r_census for p1 and distill separately (exact token accounting), the projected
full-arm rewritten-token total, and the headroom against B, with each measured r compared to the
1.5B census value so an adverse population shift is visible rather than inferred.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from kys3b.calibration import (
    DEFAULT_PILOT_SHARDS,
    REF_R,
    aggregate,
    pilot_shards,
    project_yield,
    verify_pilot,
)
from kys3b.config import PASSES, REWRITE_SETTINGS, Config
from kys3b.io import atomic_write_json, check, log
from kys3b.shards import load_index

MIN_HEADROOM_PCT = 5.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="calibration pilot report (fails closed)")
    ap.add_argument("--settings", nargs="*", default=list(REWRITE_SETTINGS))
    ap.add_argument(
        "--shards", type=int, default=DEFAULT_PILOT_SHARDS, metavar="K",
        help="the pilot size that was launched; the report REQUIRES exactly these shard ids",
    )
    ap.add_argument("--from", dest="src", default="pilot", choices=["pilot", "production"])
    args = ap.parse_args(argv)
    cfg = Config.load()
    check(8 <= args.shards <= 16, f"--shards {args.shards} is outside the 8-16 pilot band")

    blocking: dict[str, list[str]] = {}
    results: dict = {}
    header_shown = False

    print(
        f"Pilot gate: requiring all {args.shards} deterministic pilot shards, for BOTH passes, "
        f"in every requested setting.\n"
    )
    for setting in args.settings:
        src_dir = cfg.stage("sources") / setting
        idx_path = src_dir / "_shards.json"
        if not idx_path.exists():
            blocking[setting] = ["sources not materialized (no _shards.json)"]
            continue
        idx = load_index(src_dir)
        n = int(idx["n_shards"])
        expected = pilot_shards(n, args.shards)

        problems: list[str] = []
        per_pass_agg: dict[str, dict] = {}
        for pass_name in PASSES:
            base = cfg.stage("rewritten") / setting
            out_dir = (base / "_pilot" / pass_name) if args.src == "pilot" else (base / pass_name)
            probs = verify_pilot(src_dir, out_dir, expected, idx)
            if probs:
                problems += [f"{pass_name}: {x}" for x in probs]
                continue
            per_pass_agg[pass_name] = aggregate(src_dir, out_dir, expected)

        missing_passes = [pn for pn in PASSES if pn not in per_pass_agg]
        if missing_passes:
            problems.append(
                f"incomplete passes {missing_passes} -- a setting with only one pass cannot be "
                "projected and is a BLOCKER"
            )
        if problems:
            blocking[setting] = problems
            continue

        if not header_shown:
            print(
                f"{'setting':22s} {'pass':8s} {'shards':>6s} {'src_tok':>15s} {'out_tok':>15s} "
                f"{'r_census':>9s} {'r_1.5B':>8s} {'delta':>8s} {'r_status2':>10s}"
            )
            header_shown = True
        detail = {}
        for pass_name in PASSES:
            agg = per_pass_agg[pass_name]
            ref = REF_R[setting][pass_name]
            detail[pass_name] = dict(shards=expected, reference_r=ref, **agg)
            print(
                f"{setting:22s} {pass_name:8s} {len(expected):>6} "
                f"{agg['src_train_tokens_all']:>15,} {agg['out_tokens_status2']:>15,} "
                f"{agg['r_census']:>9.4f} {ref:>8.4f} "
                f"{100*(agg['r_census']-ref)/ref:>7.1f}% {agg['r_status2']:>10.4f}"
            )
        results[setting] = dict(
            expected_shards=expected,
            per_pass=detail,
            projection=project_yield(
                cfg, {k: v["r_census"] for k, v in per_pass_agg.items()},
                cfg.source_budget(setting),
            ),
        )

    # ---- blockers first, and they are fatal ----
    if blocking:
        print("\nINCOMPLETE / INVALID PILOT -- no projection will be produced for:")
        for setting, probs in blocking.items():
            print(f"\n  {setting}  ({len(probs)} problem(s))")
            for x in probs[:20]:
                print(f"    - {x}")
            if len(probs) > 20:
                print(f"    - ... and {len(probs) - 20} more")

    dest = cfg.stage("reports") / f"pilot_{args.src}.json"
    atomic_write_json(
        dict(
            source=args.src, target=cfg.B, pilot_shards=args.shards,
            complete=not blocking, blocking=blocking, results=results,
        ),
        dest,
    )
    log(f"written {dest}")

    if blocking:
        print(
            "\n  GATE FAILED: the pilot is incomplete.  Re-run the missing pilot jobs and "
            "re-run this report.  No production-go recommendation is given."
        )
        return 2
    if not results:
        print("\n  GATE FAILED: no pilot results at all -- nothing was measured.")
        return 2

    print(f"\nprojected full-arm yield vs the {cfg.B/1e9:.0f}B target:")
    print(
        f"  {'setting':22s} {'source':>8s} {'proj p1':>10s} {'proj distill':>13s} "
        f"{'proj total':>12s} {'headroom':>10s}"
    )
    tight = []
    for setting, r in results.items():
        pr = r["projection"]
        flag = "" if pr["meets_target"] else "   <-- BELOW TARGET"
        if not pr["meets_target"] or pr["headroom_pct"] < MIN_HEADROOM_PCT:
            tight.append(setting)
        print(
            f"  {setting:22s} {pr['source_budget']/1e9:7.0f}B "
            f"{pr['projected_p1_tokens']/1e9:9.2f}B {pr['projected_distill_tokens']/1e9:12.2f}B "
            f"{pr['projected_total_tokens']/1e9:11.2f}B {pr['headroom_pct']:+9.1f}%{flag}"
        )

    missing_settings = [s for s in REWRITE_SETTINGS if s not in results]
    if missing_settings:
        print(
            f"\n  NOTE: not measured this run: {', '.join(missing_settings)}.  "
            "GATE PASSED below covers only the settings reported above."
        )
    if tight:
        print(
            f"\n  GATE FAILED: {', '.join(tight)} project under {MIN_HEADROOM_PCT:.0f}% headroom "
            f"against the {cfg.B/1e9:.0f}B target.  Do NOT launch production: the arm would "
            "finish short and the corpus could not be assembled."
        )
        return 1
    print(
        f"\n  GATE PASSED: every measured arm projects above the {cfg.B/1e9:.0f}B target with "
        f"at least {MIN_HEADROOM_PCT:.0f}% margin, on the complete deterministic pilot set."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
