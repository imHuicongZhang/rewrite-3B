#!/usr/bin/env python
"""Stage 03 -- print (and optionally submit) the rewriting arrays.

The scope is configs/production.yaml, never "every rewriting setting": as of 2026-10-04 that is
disagreement-aware, quality-first and diversity-oriented, both passes.  wrap-inspired and
rewire-inspired are excluded, and asking for them is a hard error, not a silent skip.

One array per queue.  Every task walks the same ordered worklist of (setting, pass) items
(slurm/rewrite_array.sbatch), so a GPU moves to the next item when one drains rather than exiting.

By default this PRINTS the sbatch commands and submits nothing, which is the 1.5B
`launch_all.sh` contract.  `--submit` is required to actually queue work.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

import _bootstrap  # noqa: F401

from kys3b.config import Config, production_scope
from kys3b.io import check
from kys3b.shards import load_index


def worklist(scope, settings, passes) -> list[str]:
    for s in settings:
        check(
            s in scope["allowed_settings"],
            f"{s} is not in the production scope "
            f"({scope['excluded_settings'].get(s, 'not in allowed_settings')}); "
            "see configs/production.yaml",
        )
    for p in passes:
        check(p in scope["passes"], f"pass {p!r} is not in the production scope")
    # setting-major, smallest first (the 1.5B order); production.yaml fixes the order
    return [f"{s}:{p}" for s in scope["allowed_settings"] if s in settings for p in passes]


def git_head(code) -> tuple[str, int]:
    head = subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(code), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    return head, len(dirty)


def sbatch_cmd(cfg, items, queue, commit, array=None, test_only=False):
    g = cfg.cluster["gpu"]
    q = g["queues"][queue]
    logs = cfg.stage("logs")
    cmd = [
        "sbatch",
        *(["--test-only"] if test_only else []),
        f"--job-name=rw3_{queue}",
        f"--partition={g['partitions']}",
        f"--account={q['account']}",
        f"--qos={q['qos']}",
        f"--gres={g['gres']}",
        f"--cpus-per-task={g['cpus_per_task']}",
        f"--time={g['time']}",
        f"--array={array or q['array']}",
        f"--output={logs}/rewrite/%x_%A_%a.out",
    ]
    if q.get("requeue"):
        cmd.append("--requeue")
    cmd += [
        f"--export=ALL,KYS_WORKLIST={'+'.join(items)},KYS_CODE_COMMIT={commit}",
        str(cfg.root("code") / "slurm" / "rewrite_array.sbatch"),
    ]
    return cmd


def main(argv=None) -> int:
    scope = production_scope()
    ap = argparse.ArgumentParser(description="Stage 03: rewriting array launcher")
    ap.add_argument("--settings", nargs="*", default=scope["allowed_settings"],
                    help="subset of configs/production.yaml allowed_settings (default: all of them)")
    ap.add_argument("--passes", nargs="*", default=scope["passes"])
    ap.add_argument("--queue", default="primary", choices=["primary", "opportunistic", "both"])
    ap.add_argument("--primary-array", default=None, help="override cluster.yaml, e.g. 0-31%%32")
    ap.add_argument("--opportunistic-array", default=None, help="override cluster.yaml, e.g. 0-95%%96")
    ap.add_argument("--submit", action="store_true", help="actually submit (default: print only)")
    ap.add_argument("--test-only", action="store_true", help="sbatch --test-only (creates no job)")
    args = ap.parse_args(argv)

    cfg = Config.load()
    items = worklist(scope, args.settings, args.passes)
    check(bool(items), "empty worklist")
    commit, dirty = git_head(cfg.root("code"))
    if args.submit:
        check(dirty == 0, f"code checkout has {dirty} uncommitted change(s); commit before submitting")
    (cfg.stage("logs") / "rewrite").mkdir(parents=True, exist_ok=True)
    queues = ["primary", "opportunistic"] if args.queue == "both" else [args.queue]
    arrays = dict(primary=args.primary_array, opportunistic=args.opportunistic_array)

    print(f"# Production rewriting scope: {', '.join(scope['allowed_settings'])}")
    print(f"# Excluded (no new generation): {', '.join(scope['excluded_settings']) or 'none'}")
    print(f"# Code commit {commit} ({'clean' if not dirty else f'{dirty} dirty'})")
    print("# Worklist, walked in order by every task (both passes over the full source, Decision 3):")
    for it in items:
        s, p = it.split(":")
        idx_path = cfg.stage("sources") / s / "_shards.json"
        n = load_index(cfg.stage("sources") / s)["n_shards"] if idx_path.exists() else "not materialized"
        print(f"#   {s}/{p}: {n} shards")
    print()
    for queue in queues:
        cmd = sbatch_cmd(cfg, items, queue, commit, arrays[queue], test_only=args.test_only)
        print(f"# [{queue}]")
        print("  " + shlex.join(cmd) + "\n")
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
