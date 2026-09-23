#!/usr/bin/env python
"""Stage 00 -- download and verify the scored pool.  182 GB, 200 shards.

Refuses to run unless the target filesystem has headroom, and refuses to write to the
scratch quota (Decision 7: /weka/projects/bvandur1 is the physical root).
"""
from __future__ import annotations

import argparse
import shutil
import sys

import _bootstrap  # noqa: F401

from kys3b import manifest
from kys3b.config import Config
from kys3b.io import check, log
from kys3b.pool import validate_files

NEEDED_GB = 200  # 182 GB of parquet plus slack


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 00: download + verify the scored pool")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    cfg = Config.load()
    dest = cfg.root("pool")
    log(f"pool root: {dest}")

    if not args.verify_only:
        dest.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(dest).free / 2**30
        check(
            free_gb > NEEDED_GB,
            f"only {free_gb:.0f} GB free at {dest}; need > {NEEDED_GB} GB. "
            "Decision 7: the physical root is /weka/projects/bvandur1/zhuicon1/rewrite-3b.",
        )
        log(f"{free_gb:.0f} GB free -- ok")
        from huggingface_hub import snapshot_download

        repo = cfg.paths["hf"]["pool_repo"]
        rev = cfg.paths["hf"]["pool_revision"]
        log(f"downloading {repo}@{rev} -> {dest} (gated; needs your HF token)")
        path = snapshot_download(
            repo_id=repo, repo_type="dataset", revision=rev,
            local_dir=str(dest), max_workers=args.workers,
            allow_patterns=["*.parquet", "README.md", ".gitattributes"],
        )
        log(f"download complete: {path}")

    log("verifying shard geometry and schema (footers only)...")
    info = validate_files(cfg)
    log(f"OK: {info['n_shards']} shards, {info['total_rows']:,} rows")

    manifest.write(
        dest, "00_pool",
        dict(
            repo=cfg.paths["hf"]["pool_repo"],
            revision=cfg.paths["hf"]["pool_revision"],
            n_shards=info["n_shards"],
            total_rows=info["total_rows"],
            rows_per_shard=info["rows_per_shard"],
            exclusions_applied="none by this pipeline -- the published pool is used as-is "
            "(Decision 1: the 5M scorer/topic analysis sample is NOT removed; ~4,998,853 of "
            "its documents are present in the published 99,949,162-row table, exactly as in "
            "the 1.5B run)",
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
