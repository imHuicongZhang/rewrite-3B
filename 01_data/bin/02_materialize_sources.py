#!/usr/bin/env python
"""Stage 02 -- materialize per-setting rewriting input as 10,000-row (doc_id, text) shards.

`_shards.json` pins the shard geometry, and `shard_index` seeds the WRAP style assignment,
so this stage must be reproducible: the geometry is a pure function of the sorted doc_id set.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import numpy as np

from kys3b import manifest
from kys3b.config import REWRITE_SETTINGS, Config
from kys3b.io import check, log
from kys3b.shards import materialize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 02: materialize rewriting input shards")
    ap.add_argument("--settings", nargs="*", default=list(REWRITE_SETTINGS))
    ap.add_argument("--report-only", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args(argv)

    cfg = Config.load()
    sel_dir = cfg.stage("selection")
    manifest.read(sel_dir, "01_selection")
    rows_per_shard = int(cfg.budgets["sharding"]["rows_per_shard"])

    for setting in args.settings:
        check(setting in REWRITE_SETTINGS, f"{setting} is not a rewriting setting")
        blk_dir = sel_dir / setting
        m = manifest.read(blk_dir, "01_selection")
        ids = np.load(blk_dir / "doc_ids.npy")
        check(
            manifest.sha256_ids(ids) == m["doc_ids_sha256"],
            f"{setting}: doc_ids.npy digest does not match its manifest",
        )
        n_shards = int(np.ceil(ids.size / rows_per_shard))
        log(f"{setting}: {ids.size:,} docs -> {n_shards} shards of {rows_per_shard}")
        if args.report_only:
            continue
        out_dir = cfg.stage("sources") / setting
        index = materialize(cfg, setting, ids, out_dir, rows_per_shard)
        manifest.write(
            out_dir, "02_sources",
            dict(
                setting=setting, rows_per_shard=rows_per_shard,
                n_shards=index["n_shards"], total_docs=index["total_docs"],
                selection_doc_ids_sha256=m["doc_ids_sha256"],
                selection_train_tokens=m["train_tokens"],
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
