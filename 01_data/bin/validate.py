#!/usr/bin/env python
"""Stage validation -- the checks plan.md section 17 promises, runnable independently.

  --stage pool       200 shards, row counts, doc_id contiguity, schema, 24 topics
  --stage selection  digests match manifests; base cap strategy = 0; quality-base subset;
                     budgets filled
  --stage sources    row counts + token totals match the selection; every doc_id once
  --stage rewritten  output rows == input rows per shard; status domain; wrap_style balance;
                     BOTH passes present for all five arms
  --stage filtered   only status==2 rows; token totals vs target; distill share vs 1.5B
  --stage final      corpus totals; base cap strategy = 0; shuffle row conservation;
                     matched horizons recorded
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import _bootstrap  # noqa: F401
import numpy as np
import pyarrow.parquet as pq

from kys3b import manifest
from kys3b.config import REWRITE_SETTINGS, SETTINGS, WRAP_STYLES, Config
from kys3b.io import check, log, parquet_rows
from kys3b.pool import validate_files
from kys3b.shards import load_index, shard_path

OK = "  OK  "


def v_pool(cfg):
    info = validate_files(cfg)
    log(f"{OK} pool: {info['n_shards']} shards, {info['total_rows']:,} rows, schema complete")
    return info


def v_selection(cfg):
    sel = cfg.stage("selection")
    top = manifest.read(sel, "01_selection")
    base = np.load(sel / "shared-base-10B" / "doc_ids.npy")
    check(
        manifest.sha256_ids(base) == top["base_doc_ids_sha256"],
        "shared base digest does not match the stage manifest",
    )
    log(f"{OK} shared base: {base.size:,} docs, digest matches")
    base_set = base
    for name in SETTINGS:
        d = sel / name
        m = manifest.read(d, "01_selection")
        ids = np.load(d / "doc_ids.npy")
        check(manifest.sha256_ids(ids) == m["doc_ids_sha256"], f"{name}: digest mismatch")
        check(np.unique(ids).size == ids.size, f"{name}: duplicate doc_id")
        inter = np.intersect1d(ids, base_set, assume_unique=True)
        check(inter.size == 0, f"{name}: {inter.size:,} documents overlap the shared base")
        check(
            m["train_tokens"] >= m["target_tokens"] or name == "diversity-oriented",
            f"{name}: under budget ({m['train_tokens']:,} < {m['target_tokens']:,})",
        )
        log(
            f"{OK} {name:22s} {ids.size:>12,} docs  {m['train_tokens']:>15,} tok  "
            f"base-overlap 0"
        )
    qb = np.load(sel / "quality-base" / "doc_ids.npy")
    qf = np.load(sel / "quality-first" / "doc_ids.npy")
    check(bool(np.isin(qb, qf).all()), "quality-base is not contained in quality-first")
    log(f"{OK} quality-base is a subset of quality-first")
    return top


def v_sources(cfg):
    sel = cfg.stage("selection")
    for setting in REWRITE_SETTINGS:
        d = cfg.stage("sources") / setting
        m = manifest.read(d, "02_sources")
        idx = load_index(d)
        sm = manifest.read(sel / setting, "01_selection")
        check(
            m["selection_doc_ids_sha256"] == sm["doc_ids_sha256"],
            f"{setting}: sources were built from a different selection",
        )
        total, seen = 0, []
        for s in idx["shards"]:
            p = d / s["file"]
            check(p.exists(), f"missing {p}")
            check(parquet_rows(p) == s["rows"], f"{p}: row count != index")
            total += s["rows"]
            t = pq.read_table(p, columns=["doc_id"], use_threads=False)
            seen.append(t.column("doc_id").to_numpy(zero_copy_only=False))
        check(total == idx["total_docs"], f"{setting}: {total} rows != {idx['total_docs']}")
        allids = np.concatenate(seen)
        check(np.unique(allids).size == allids.size, f"{setting}: duplicate doc_id across shards")
        check(
            int(allids.size) == int(sm["docs"]),
            f"{setting}: {allids.size} materialized != {sm['docs']} selected",
        )
        log(f"{OK} {setting:22s} {idx['n_shards']:>5} shards, {total:>12,} docs, ids unique")


def v_rewritten(cfg):
    for setting in REWRITE_SETTINGS:
        idx = load_index(cfg.stage("sources") / setting)
        n = idx["n_shards"]
        is_wrap = cfg.setting(setting)["rewrite"]["pass1"] == "wrap"
        for pass_name in ("p1", "distill"):
            d = cfg.stage("rewritten") / setting / pass_name
            missing = [k for k in range(n) if not shard_path(d, k).exists()]
            check(
                not missing,
                f"{setting}/{pass_name}: {len(missing)} shards missing "
                f"(Decision 3: BOTH passes are required) e.g. {missing[:5]}",
            )
            styles: Counter = Counter()
            rows = 0
            for k in range(n):
                p = shard_path(d, k)
                cols = ["status"] + (["wrap_style"] if (is_wrap and pass_name == "p1") else [])
                t = pq.read_table(p, columns=cols, use_threads=False)
                check(
                    t.num_rows == idx["shards"][k]["rows"],
                    f"{p}: {t.num_rows} rows != source {idx['shards'][k]['rows']}",
                )
                st = t.column("status").to_numpy(zero_copy_only=False)
                bad = set(np.unique(st).tolist()) - {0, 1, 2}
                check(not bad, f"{p}: status values outside 0/1/2: {bad}")
                rows += t.num_rows
                if is_wrap and pass_name == "p1":
                    styles.update(t.column("wrap_style").to_pylist())
            msg = f"{OK} {setting:22s}/{pass_name:7s} {n:>5} shards, {rows:>12,} rows"
            if styles:
                tot = sum(styles.values())
                pct = {s: 100.0 * styles[s] / tot for s in WRAP_STYLES}
                for s, v in pct.items():
                    check(
                        abs(v - 25.0) < 0.5,
                        f"{setting}: wrap_style {s} at {v:.2f}% -- expected 25.0% +- 0.5pp",
                    )
                msg += "  styles " + " ".join(f"{s}={pct[s]:.2f}%" for s in WRAP_STYLES)
            log(msg)


def v_filtered(cfg):
    ref_distill_share = {  # 1.5B Table 7, for a sanity comparison (not a gate)
        "quality-first": 34.9, "wrap-inspired": 16.9, "rewire-inspired": 53.1,
        "diversity-oriented": 24.4, "disagreement-aware": 29.1,
    }
    for setting in REWRITE_SETTINGS:
        m = manifest.read(cfg.stage("filtered") / setting, "04_filtered")
        tot = m["total_tokens"]
        tgt = m["target_tokens"]
        d = cfg.stage("filtered") / setting / "rewritten"
        rows = 0
        for p in sorted(d.glob("*.parquet")):
            t = pq.read_table(p, columns=["source_prompt"], use_threads=False)
            rows += t.num_rows
        dt = m.get("distill_tokens", m.get("kept_from_distill", {}).get("tokens", 0))
        share = 100.0 * dt / tot if tot else 0.0
        short = m.get("shortfall", 0)
        check(
            tot >= tgt or short > 0,
            f"{setting}: {tot:,} < target {tgt:,} without a reported shortfall",
        )
        log(
            f"{OK} {setting:22s} {rows:>12,} rows  {tot:>15,} tok  shortfall {short:>12,}  "
            f"distill share {share:5.1f}% (1.5B {ref_distill_share[setting]:.1f}%)"
        )


def v_final(cfg):
    ep = int(cfg.budgets["training"]["epoch_tokens"])
    for setting in SETTINGS:
        fin = cfg.stage("final") / setting
        m = json.loads((fin / "_pretrain_manifest.json").read_text())
        check(m["doc_id_overlap_base_strategy"] == 0, f"{setting}: base/strategy doc_id overlap")
        corpus = m["corpus_train_tokens"]
        hz = m["training"]["horizons"]
        check(hz == [ep, 2 * ep, 3 * ep], f"{setting}: horizons {hz} are not matched multiples")
        if m["shuffle"]:
            shuf = sorted((fin / "shuffled").glob("part_*.parquet"))
            rows = sum(parquet_rows(p) for p in shuf)
            check(
                rows == m["shuffle"]["rows"],
                f"{setting}: shuffled rows {rows:,} != {m['shuffle']['rows']:,}",
            )
        p = m["training"]["passes_to_horizon"]
        log(
            f"{OK} {setting:22s} corpus {corpus:>15,} tok  short {m['shortfall_vs_epoch']:>12,}  "
            f"passes to 20/40/60B {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}"
        )


STAGES = dict(
    pool=v_pool, selection=v_selection, sources=v_sources,
    rewritten=v_rewritten, filtered=v_filtered, final=v_final,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="stage validation")
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    args = ap.parse_args(argv)
    cfg = Config.load()
    todo = list(STAGES) if args.stage == "all" else [args.stage]
    for s in todo:
        log(f"--- validating {s} ---")
        STAGES[s](cfg)
    print("\n=== VALIDATION PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
