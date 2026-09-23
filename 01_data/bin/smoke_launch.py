#!/usr/bin/env python
"""Bounded GPU smoke test: one task, one shard, a handful of documents.

Exercises every distinct worker code path before any production GPU time:

  grounded Wikipedia pass   quality-first / p1
  distill pass              quality-first / distill
  WRAP 4-style pass         wrap-inspired / p1
  REWIRE both pass inputs   rewire-inspired / p1  and  rewire-inspired / distill

The other grounded arms (diversity-oriented, disagreement-aware) share the identical worker path
with quality-first and are not smoke-tested separately.

Like `03_rewrite_launch.py`, this PRINTS and submits nothing unless `--submit` is given.  With
`--smoke-rows` set (the default) output is written to `<setting>/_smoke/<pass>/`, never the
production path, so a smoke run can never make a later production job skip data.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from kys3b.config import Config
from kys3b.io import check

# (setting, pass, what it exercises)
SMOKE_MATRIX = [
    ("quality-first", "p1", "grounded Wikipedia-style pass"),
    ("quality-first", "distill", "distill pass"),
    ("wrap-inspired", "p1", "WRAP 4-style per-document assignment"),
    ("rewire-inspired", "p1", "REWIRE pass-1 input"),
    ("rewire-inspired", "distill", "REWIRE distill input"),
]


def sbatch_cmd(cfg, setting, pass_name, rows, partition, test_only=False):
    g = cfg.cluster["gpu"]
    q = g["queues"]["primary"]
    logs = cfg.stage("logs")
    return [
        "sbatch",
        *(["--test-only"] if test_only else []),
        f"--job-name=smoke_{setting}_{pass_name}",
        f"--partition={partition}",
        f"--account={q['account']}",
        f"--qos={q['qos']}",
        f"--gres={g['gres']}",
        f"--cpus-per-task={g['cpus_per_task']}",
        "--time=01:00:00",
        f"--output={logs}/smoke/%x_%j.out",
        f"--export=ALL,KYS_SETTING={setting},KYS_PASS={pass_name},KYS_SMOKE_ROWS={rows}",
        str(cfg.root("code") / "slurm" / "smoke.sbatch"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="bounded GPU smoke test launcher")
    ap.add_argument("--smoke-rows", type=int, default=64)
    ap.add_argument("--partition", default=None, help="default: cluster.yaml gpu.partitions")
    ap.add_argument("--only", nargs="*", default=None, help="e.g. quality-first/p1")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--test-only", action="store_true", help="sbatch --test-only, creates no job")
    args = ap.parse_args(argv)

    cfg = Config.load()
    check(args.smoke_rows > 0, "--smoke-rows must be positive")
    check(
        args.smoke_rows <= 2000,
        f"--smoke-rows {args.smoke_rows} is not a smoke test; use 03_rewrite_launch.py "
        "for production",
    )
    partition = args.partition or cfg.cluster["gpu"]["partitions"]
    (cfg.stage("logs") / "smoke").mkdir(parents=True, exist_ok=True)

    print(
        f"# Bounded GPU smoke: 1 task, 1 shard, {args.smoke_rows} documents per job.\n"
        f"# Output -> <setting>/_smoke/<pass>/ (NEVER the production path), so production still\n"
        f"# sees every shard as outstanding.\n"
    )
    rc = 0
    for setting, pass_name, what in SMOKE_MATRIX:
        if args.only and f"{setting}/{pass_name}" not in args.only:
            continue
        cmd = sbatch_cmd(cfg, setting, pass_name, args.smoke_rows, partition, args.test_only)
        print(f"# {setting}/{pass_name}  --  {what}")
        print("  " + " ".join(cmd) + "\n")
        if args.submit or args.test_only:
            r = subprocess.run(cmd, capture_output=True, text=True)
            for line in (r.stdout + r.stderr).splitlines():
                if line.strip() and "cli_filter" not in line and "BILLING" not in line:
                    print("    " + line)
            rc = rc or r.returncode
    if not (args.submit or args.test_only):
        print("# Nothing submitted.  --test-only validates; --submit queues.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
