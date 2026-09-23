#!/usr/bin/env python
"""Stage 05 -- materialize the shared base, mix in the strategy half, shuffle.

`05_final/<setting>/shuffled/` is THE DELIVERABLE of this repository.  Tokenization into
Nanotron's format is the training repo's job (Decision 10).

Per setting:
  1. write shared-base-10B/ from the pool (original text, source_prompt="original")
  2. write strategy/ -- raw quality-base docs for QUALITY-BASE, else copy 04_filtered/
  3. assert 0 doc_id overlap between the base and the strategy half's SOURCE documents
  4. document-level shuffle, seed 42 (io.bucketed_shuffle)
  5. write _pretrain_manifest.json, including the MATCHED 20B/40B/60B token horizons and the
     pass counts each setting needs to reach them
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pyarrow.parquet as pq

from kys3b import manifest
from kys3b.config import SETTINGS, Config
from kys3b.io import atomic_write_json, atomic_write_table, check, log, parquet_rows
from kys3b.post.mix import base_table, check_no_overlap, horizon_plan, shuffle_final
from kys3b.pool import doc_id_to_shard

ROWS_PER_RAW_SHARD = 200_000


def _write_raw_block(cfg, doc_ids: np.ndarray, out_dir, label: str) -> dict:
    """Materialize original pool text for a raw block (shared base / quality-base)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = np.sort(np.asarray(doc_ids, np.int64))
    groups = [ids[i : i + ROWS_PER_RAW_SHARD] for i in range(0, ids.size, ROWS_PER_RAW_SHARD)]
    total_tokens = 0
    for gi, g in enumerate(groups):
        dest = out_dir / f"part_{gi:05d}.parquet"
        if dest.exists() and parquet_rows(dest) == g.size:
            t = pq.read_table(dest, columns=["tokens_llama2"], use_threads=False)
            total_tokens += int(
                t.column("tokens_llama2").to_numpy(zero_copy_only=False).astype(np.int64).sum()
                + g.size
            )
            continue
        psh, prow = doc_id_to_shard(cfg, g)
        texts: dict[int, str] = {}
        toks: dict[int, int] = {}
        for ps in np.unique(psh):
            m = psh == ps
            t = pq.read_table(
                cfg.pool_shard(int(ps)), columns=["doc_id", "text", "tokens-llama2"],
                use_threads=False,
            )
            pool_ids = t.column("doc_id").to_numpy(zero_copy_only=False)
            tx = t.column("text").to_pylist()
            tk = t.column("tokens-llama2").to_numpy(zero_copy_only=False)
            rows = prow[m]
            check(bool(np.array_equal(pool_ids[rows], g[m])), f"{label}: doc_id mismatch")
            for did, r in zip(g[m].tolist(), rows.tolist()):
                texts[did] = tx[r]
                toks[did] = int(tk[r])
            del t, tx
        tbl = base_table(g, [texts[int(d)] for d in g], [toks[int(d)] for d in g])
        atomic_write_table(tbl, dest)
        total_tokens += int(sum(toks[int(d)] for d in g) + g.size)  # +1 BOS per document
    return dict(docs=int(ids.size), train_tokens=int(total_tokens), shards=len(groups))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 05: base + strategy -> shuffled corpus")
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS))
    ap.add_argument("--mem-gb", type=float, default=240.0)
    ap.add_argument("--skip-shuffle", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.load()
    sel_dir = cfg.stage("selection")
    manifest.read(sel_dir, "01_selection")
    base_ids = np.load(sel_dir / "shared-base-10B" / "doc_ids.npy")
    out = {}

    for setting in args.settings:
        log(f"=== final assembly: {setting} ===")
        fin = cfg.stage("final") / setting
        base_dir = fin / "shared-base-10B"
        strat_dir = fin / "strategy"

        b = _write_raw_block(cfg, base_ids, base_dir, "shared base")
        log(f"  base: {b['docs']:,} docs, {b['train_tokens']:,} train tokens")

        if setting == "quality-base":
            qb_ids = np.load(sel_dir / "quality-base" / "doc_ids.npy")
            s = _write_raw_block(cfg, qb_ids, strat_dir, "quality-base")
            strat_source_ids = qb_ids
            strat_paths = sorted(strat_dir.glob("part_*.parquet"))
        else:
            src = cfg.stage("filtered") / setting / "rewritten"
            manifest.read(cfg.stage("filtered") / setting, "04_filtered")
            strat_dir.mkdir(parents=True, exist_ok=True)
            paths = sorted(src.glob("*.parquet"))
            check(bool(paths), f"{setting}: no assembled shards at {src}")
            ids_seen, tokens = [], 0
            for p in paths:
                dest = strat_dir / p.name
                t = pq.read_table(p, use_threads=False)
                if not (dest.exists() and parquet_rows(dest) == t.num_rows):
                    atomic_write_table(t, dest)
                ids_seen.append(t.column("orig_doc_id").to_numpy(zero_copy_only=False))
                tokens += int(
                    t.column("tokens_llama2").to_numpy(zero_copy_only=False).astype(np.int64).sum()
                    + t.num_rows
                )
            strat_source_ids = np.concatenate(ids_seen)
            s = dict(docs=int(strat_source_ids.size), train_tokens=int(tokens), shards=len(paths))
            strat_paths = sorted(strat_dir.glob("*.parquet"))
        log(f"  strategy: {s['docs']:,} rows, {s['train_tokens']:,} train tokens")

        # the 1.5B Step-3 gate
        check_no_overlap(base_ids, strat_source_ids)
        log("  doc_id overlap base <-> strategy sources: 0 (asserted)")

        corpus_tokens = b["train_tokens"] + s["train_tokens"]
        shuf = None
        if not args.skip_shuffle:
            shuf = shuffle_final(
                sorted(base_dir.glob("part_*.parquet")) + list(strat_paths),
                fin / "shuffled", fin / "_shuffle_tmp",
                seed=cfg.seed, mem_gb=args.mem_gb,
            )
            log(f"  shuffled: {shuf['rows']:,} rows -> {shuf['out_shards']} shards")

        plan = horizon_plan(corpus_tokens, cfg)
        payload = dict(
            setting=setting,
            base=b, strategy=s,
            corpus_train_tokens=int(corpus_tokens),
            epoch_target=int(cfg.budgets["training"]["epoch_tokens"]),
            shortfall_vs_epoch=int(max(0, cfg.budgets["training"]["epoch_tokens"] - corpus_tokens)),
            doc_id_overlap_base_strategy=0,
            shuffle=shuf,
            training=plan,
        )
        atomic_write_json(payload, fin / "_pretrain_manifest.json")
        manifest.write(fin, "05_final", payload)
        out[setting] = payload

    atomic_write_json(out, cfg.stage("reports") / "final_corpora.json")
    log("")
    log("=== matched training-token horizons (Decision 5) ===")
    for k, v in out.items():
        p = v["training"]["passes_to_horizon"]
        log(
            f"  {k:22s} corpus {v['corpus_train_tokens']:,} tok -> passes to 20/40/60B: "
            f"{p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
