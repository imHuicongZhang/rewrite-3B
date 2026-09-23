#!/usr/bin/env python
"""Live progress: per (setting, pass) shard completion, claim state, and token yield."""
from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401

from kys3b.claims import ClaimDir
from kys3b.config import PASSES, REWRITE_SETTINGS, Config
from kys3b.io import parquet_rows
from kys3b.shards import load_index, shard_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="rewriting progress")
    ap.add_argument("--settings", nargs="*", default=list(REWRITE_SETTINGS))
    args = ap.parse_args(argv)
    cfg = Config.load()
    stale = int(cfg.budgets["sharding"]["claim_stale_seconds"])

    hdr = f"{'setting':22s} {'pass':8s} {'shards':>8s} {'done':>8s} {'run':>5s} {'stale':>6s} {'pend':>7s} {'out_tok':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for setting in args.settings:
        try:
            idx = load_index(cfg.stage("sources") / setting)
        except Exception:
            print(f"{setting:22s} (sources not materialized)")
            continue
        n = idx["n_shards"]
        for pass_name in PASSES:
            d = cfg.stage("rewritten") / setting / pass_name

            def done(s, d=d, idx=idx):
                p = shard_path(d, s)
                if not p.exists():
                    return False
                try:
                    return parquet_rows(p) == idx["shards"][s]["rows"]
                except Exception:
                    return False

            cl = ClaimDir(cfg.stage("rewritten") / setting / "_claims" / pass_name, stale)
            st = cl.status(range(n), done)
            tok = 0
            pdir = cfg.stage("rewritten") / setting / "_progress" / pass_name
            if pdir.exists():
                for f in pdir.glob("worker_*.json"):
                    try:
                        tok += int(json.loads(f.read_text()).get("total_output_tokens", 0))
                    except Exception:
                        pass
            print(
                f"{setting:22s} {pass_name:8s} {n:>8,} {st['done']:>8,} {st['in_flight']:>5} "
                f"{st['stale']:>6} {st['pending']:>7,} {tok:>16,}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
