#!/usr/bin/env python
"""Calibration pilot: a few COMPLETE shards per arm, spread across the whole corpus.

The 64-row smoke test proves the machinery works; it cannot tell you whether the rewritten-output
yield clears the 10B target.  QUALITY-FIRST is projected at only +19.6% headroom (plan.md 18.2),
so this gate runs real, complete shards and measures r exactly before thousands of GPU-hours.

Shards are the K mid-quantile indices of the shard range, never the first K: source shards are
contiguous slices of a doc_id-sorted selection, so a leading block is a systematically different
slice of the corpus.

Prints and submits nothing unless --submit.  Read the result with `bin/pilot_report.py`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from kys3b.calibration import DEFAULT_PILOT_SHARDS, pilot_shards
from kys3b.config import PASSES, REWRITE_SETTINGS, Config
from kys3b.io import check
from kys3b.shards import load_index


def sbatch_cmd(cfg, setting, pass_name, k, to_production, test_only=False):
    g = cfg.cluster["gpu"]
    q = g["queues"]["primary"]
    exports = [
        "ALL",
        f"KYS_SETTING={setting}",
        f"KYS_PASS={pass_name}",
        f"KYS_PILOT_K={k}",
    ]
    if to_production:
        exports.append("KYS_PILOT_TO_PRODUCTION=1")
    return [
        "sbatch",
        *(["--test-only"] if test_only else []),
        f"--job-name=pilot_{setting}_{pass_name}",
        f"--partition={g['partitions']}",
        f"--account={q['account']}",
        f"--qos={q['qos']}",
        f"--gres={g['gres']}",
        f"--cpus-per-task={g['cpus_per_task']}",
        "--time=12:00:00",
        f"--output={cfg.stage('logs')}/pilot/%x_%j.out",
        "--export=" + ",".join(exports),
        str(cfg.root("code") / "slurm" / "pilot.sbatch"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="calibration pilot launcher")
    ap.add_argument("--shards", type=int, default=DEFAULT_PILOT_SHARDS, metavar="K")
    ap.add_argument("--settings", nargs="*", default=list(REWRITE_SETTINGS))
    ap.add_argument("--passes", nargs="*", default=list(PASSES))
    ap.add_argument(
        "--to-production", action="store_true",
        help="write to the production path so the work counts as finished (default: _pilot/)",
    )
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--test-only", action="store_true", help="sbatch --test-only, creates no job")
    args = ap.parse_args(argv)

    cfg = Config.load()
    check(
        8 <= args.shards <= 16,
        f"--shards {args.shards} is outside the intended 8-16 band for a pilot",
    )
    (cfg.stage("logs") / "pilot").mkdir(parents=True, exist_ok=True)

    print(
        f"# Calibration pilot: {args.shards} COMPLETE shards per (setting, pass), evenly spread\n"
        f"# across the whole shard range.  Production generation semantics.  Output -> "
        f"{'PRODUCTION path' if args.to_production else '<setting>/_pilot/<pass>/'}.\n"
    )
    rc = 0
    for setting in args.settings:
        check(setting in REWRITE_SETTINGS, f"{setting} is not a rewriting setting")
        # quiet probe: before stage 02 there is no shard index, and load_index would print a
        # STOP banner that reads like a failure
        idx_path = cfg.stage("sources") / setting / "_shards.json"
        if idx_path.exists():
            n = int(load_index(cfg.stage("sources") / setting)["n_shards"])
            ids = pilot_shards(n, args.shards)
            span = f"{len(ids)} of {n} shards: {ids}"
        else:
            span = "sources not materialized yet (shard ids resolve at run time)"
        print(f"# {setting}  --  {span}")
        for pass_name in args.passes:
            cmd = sbatch_cmd(cfg, setting, pass_name, args.shards, args.to_production,
                             args.test_only)
            print(f"#   {pass_name}")
            print("    " + " ".join(cmd))
            if args.submit or args.test_only:
                r = subprocess.run(cmd, capture_output=True, text=True)
                for line in (r.stdout + r.stderr).splitlines():
                    if line.strip() and "cli_filter" not in line and "BILLING" not in line:
                        print("      " + line)
                rc = rc or r.returncode
        print()
    if not (args.submit or args.test_only):
        print("# Nothing submitted.  --test-only validates; --submit queues.")
        print("# After it finishes:  bin/pilot_report.py")
    return rc


if __name__ == "__main__":
    sys.exit(main())
