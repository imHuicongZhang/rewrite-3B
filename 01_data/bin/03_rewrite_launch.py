#!/usr/bin/env python
"""Stage 03 -- print (and optionally submit) the rewriting arrays.  10 jobs = 5 settings x 2 passes.

By default this PRINTS the sbatch commands and submits nothing, which is the 1.5B
`launch_all.sh` contract.  `--submit` is required to actually queue work.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import _bootstrap  # noqa: F401

from kys3b.config import PASSES, REWRITE_SETTINGS, Config
from kys3b.io import check
from kys3b.shards import load_index

# smallest first: problems surface cheaply (the 1.5B order)
ORDER = [
    "disagreement-aware",
    "quality-first",
    "diversity-oriented",
    "wrap-inspired",
    "rewire-inspired",
]


def sbatch_cmd(cfg, setting, pass_name, queue, test_only=False):
    g = cfg.cluster["gpu"]
    q = g["queues"][queue]
    root = cfg.root("code")
    logs = cfg.stage("logs")
    cmd = [
        "sbatch",
        *(["--test-only"] if test_only else []),
        f"--job-name=rw_{setting}_{pass_name}_{queue}",
        f"--partition={g['partitions']}",
        f"--account={q['account']}",
        f"--qos={q['qos']}",
        f"--gres={g['gres']}",
        f"--cpus-per-task={g['cpus_per_task']}",
        f"--time={g['time']}",
        f"--array={q['array']}",
        f"--output={logs}/rewrite/%x_%A_%a.out",
    ]
    if q.get("requeue"):
        cmd.append("--requeue")
    cmd += [
        f"--export=ALL,KYS_SETTING={setting},KYS_PASS={pass_name}",
        str(root / "slurm" / "rewrite_array.sbatch"),
    ]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 03: rewriting array launcher")
    ap.add_argument("--settings", nargs="*", default=ORDER)
    ap.add_argument("--passes", nargs="*", default=list(PASSES))
    ap.add_argument("--queue", default="primary", choices=["primary", "opportunistic", "both"])
    ap.add_argument("--submit", action="store_true", help="actually submit (default: print only)")
    ap.add_argument("--test-only", action="store_true", help="sbatch --test-only (creates no job)")
    args = ap.parse_args(argv)

    cfg = Config.load()
    (cfg.stage("logs") / "rewrite").mkdir(parents=True, exist_ok=True)
    queues = ["primary", "opportunistic"] if args.queue == "both" else [args.queue]

    print("# Rewriting plan -- 5 settings x 2 passes.  Both passes are ALWAYS generated")
    print("# over the full selected source set (Decision 3).\n")
    for setting in args.settings:
        check(setting in REWRITE_SETTINGS, f"{setting} is not a rewriting setting")
        # quiet probe: before stage 02 has run there is simply no shard count to show, and
        # load_index would print a STOP banner that reads like a failure
        idx_path = cfg.stage("sources") / setting / "_shards.json"
        n = load_index(cfg.stage("sources") / setting)["n_shards"] if idx_path.exists() else "not materialized"
        for pass_name in args.passes:
            for queue in queues:
                cmd = sbatch_cmd(cfg, setting, pass_name, queue, test_only=args.test_only)
                print(f"# {setting}/{pass_name} [{queue}] -- {n} shards")
                print("  " + " ".join(cmd) + "\n")
                if args.submit or args.test_only:
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    for line in (r.stdout + r.stderr).splitlines():
                        if line.strip() and "cli_filter" not in line and "BILLING" not in line:
                            print("    " + line)
                    if r.returncode != 0:
                        return r.returncode
    if not (args.submit or args.test_only):
        print("# Nothing submitted.  Re-run with --submit to queue, or --test-only to validate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
