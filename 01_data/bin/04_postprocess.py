#!/usr/bin/env python
"""Stage 04 -- strip, then assemble ~B rewritten tokens per setting.

Steps are separate subcommands on purpose.  The 1.5B run hit NODE_FAIL twice when a heavy
96-worker `srun` was followed by a second `srun` in the same batch job (jobs 1593189,
1610294); the documented mitigation is one job per step with `--dependency=afterok`.

  --step strip        in-place preamble strip + llama-2 recount, per (setting, pass)
  --step score-rewire REWIRE only: pool BOTH passes, fastText-score the REWRITTEN text
  --step assemble     build 04_filtered/<setting>/rewritten/ (~B train tokens)
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kys3b import manifest
from kys3b.config import REWRITE_SETTINGS, Config, env_offline
from kys3b.io import atomic_write_json, atomic_write_table, check, log
from kys3b.post import assemble as asm
from kys3b.post import rewire as rw
from kys3b.post.collect import attach_keys, collect_pass, cross_pass_report
from kys3b.post.mix import FINAL_SCHEMA
from kys3b.post.strip import strip_shard
from kys3b.shards import load_index, shard_path

PASS_DIR = {"p1": "p1", "distill": "distill"}


def _pass_dirs(cfg, setting):
    base = cfg.stage("rewritten") / setting
    return base / "p1", base / "distill"


def _n_shards(cfg, setting):
    return int(load_index(cfg.stage("sources") / setting)["n_shards"])


# --------------------------------------------------------------------------- strip
def step_strip(cfg, settings, workers):
    from kys3b.tokens import load_llama2

    ltok = load_llama2(
        str(cfg.model("llama2")), int(cfg.vllm["llama2_tokenizer"]["expected_vocab_size"])
    )
    out = {}
    for setting in settings:
        n = _n_shards(cfg, setting)
        out[setting] = {}
        for pass_name in ("p1", "distill"):
            d = cfg.stage("rewritten") / setting / PASS_DIR[pass_name]
            agg = dict(n_status2=0, n_stripped=0, tok_before_s2=0, tok_after_s2=0, tok_saved=0)
            for k in range(n):
                p = shard_path(d, k)
                check(p.exists(), f"missing {p} -- pass not complete")
                r = strip_shard(p, setting, pass_name, ltok)
                for key in agg:
                    agg[key] += r[key]
            pct = 100.0 * agg["n_stripped"] / max(1, agg["n_status2"])
            log(
                f"{setting}/{pass_name}: status2={agg['n_status2']:,} "
                f"stripped={agg['n_stripped']:,} ({pct:.3f}%) tok_saved={agg['tok_saved']:,}"
            )
            agg["stripped_pct"] = pct
            out[setting][pass_name] = agg
            manifest.write(d, "04_strip", dict(setting=setting, pass_name=pass_name, **agg))
    atomic_write_json(out, cfg.stage("reports") / "step1_strip.json")
    return out


# --------------------------------------------------------------------------- rewire scoring
def step_score_rewire(cfg, sample_ref_shards=None):
    from kys3b.fasttext_score import FastTextScorer, v2_percentile

    setting = "rewire-inspired"
    n = _n_shards(cfg, setting)
    p1_dir, d_dir = _pass_dirs(cfg, setting)
    # Decision 3 gate: both passes must be complete before any filtering
    rw.require_both_passes(
        sum(shard_path(p1_dir, k).exists() for k in range(n)),
        sum(shard_path(d_dir, k).exists() for k in range(n)),
        n,
    )
    scorer = FastTextScorer(cfg.model("fasttext"))
    ref, ref_n = rw.reference_distribution(cfg, sample_shards=sample_ref_shards)
    out_dir = cfg.stage("filtered") / setting / "scored_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"scoring the REWRITTEN text of both passes across {n} shards -> {out_dir}")

    for pass_name, src in (("p1", p1_dir), ("distill", d_dir)):
        for k in range(n):
            dest = out_dir / f"{pass_name}_{k:05d}.parquet"
            if dest.exists():
                continue
            t = pq.read_table(shard_path(src, k), use_threads=False)
            texts = t.column("rewritten").to_pylist()
            raw = scorer.score_many(texts)
            pct = v2_percentile(raw, ref, ref_n)
            atomic_write_table(
                pa.table(
                    {
                        "doc_id": t.column("doc_id"),
                        "status": t.column("status"),
                        "rewritten_tokens": t.column("rewritten_tokens"),
                        "rewritten_fasttext_score": pa.array(raw, type=pa.float32()),
                        "rewritten_fasttext_ranking_v2": pa.array(pct, type=pa.float32()),
                    }
                ),
                dest,
            )
        log(f"  {pass_name}: {n} shards scored")
    manifest.write(
        out_dir, "04_score_rewire",
        dict(
            setting=setting, n_shards=n, reference_n=ref_n,
            fasttext_model=str(cfg.model("fasttext")),
            score_definition="raw = p if __label__hq else 1-p; clean = text.replace(newline,' ')"
            ".replace(cr,' ')[:100000]; empty -> 0.0  (exact 1.5B recipe)",
            sorting="RAW score DESC (the v2 percentile is recorded for readability only)",
            both_passes_in_pool=True,
        ),
    )


# --------------------------------------------------------------------------- assemble
def _load_scored(cfg, n):
    d = cfg.stage("filtered") / "rewire-inspired" / "scored_pool"
    out = {}
    for pass_name in ("p1", "distill"):
        ids, st, tk, raw = [], [], [], []
        for k in range(n):
            p = d / f"{pass_name}_{k:05d}.parquet"
            check(p.exists(), f"missing {p} -- run --step score-rewire first")
            t = pq.read_table(p, use_threads=False)
            ids.append(t.column("doc_id").to_numpy(zero_copy_only=False))
            st.append(t.column("status").to_numpy(zero_copy_only=False).astype(np.int8))
            tk.append(
                t.column("rewritten_tokens").to_numpy(zero_copy_only=False).astype(np.int64)
            )
            raw.append(
                t.column("rewritten_fasttext_score").to_numpy(zero_copy_only=False).astype(
                    np.float32
                )
            )
        out[pass_name] = dict(
            doc_id=np.concatenate(ids),
            status=np.concatenate(st),
            rewritten_tokens=np.concatenate(tk),
            score=np.concatenate(raw),
        )
        out[pass_name]["len"] = out[pass_name]["rewritten_tokens"] + 1
    return out


def _write_assembled(cfg, setting, keep, n_shards, styles_for_p1):
    """Emit the kept rows as FINAL_SCHEMA parquet, ready for stage 05."""
    out_dir = cfg.stage("filtered") / setting / "rewritten"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = cfg.stage("rewritten") / setting
    written = {"p1": 0, "distill": 0}
    for pass_name, mask in (("p1", keep["keep_p1"]), ("distill", keep["keep_distill"])):
        off = 0
        for k in range(n_shards):
            p = shard_path(base / PASS_DIR[pass_name], k)
            t = pq.read_table(p, use_threads=False)
            n = t.num_rows
            sel = mask[off : off + n]
            off += n
            dest = out_dir / f"{pass_name}_{k:05d}.parquet"
            cnt = int(sel.sum())
            if cnt == 0:
                if dest.exists():
                    dest.unlink()
                continue
            sub = t.filter(pa.array(sel))
            if pass_name == "p1" and styles_for_p1:
                sp = [f"wrap_{s}" for s in sub.column("wrap_style").to_pylist()]
            elif pass_name == "p1":
                sp = ["wikipedia"] * cnt
            else:
                sp = ["distill"] * cnt
            atomic_write_table(
                pa.table(
                    {
                        "orig_doc_id": sub.column("doc_id"),
                        "text": sub.column("rewritten"),
                        "source_prompt": pa.array(sp, type=pa.large_string()),
                        "tokens_llama2": sub.column("rewritten_tokens"),
                    },
                    schema=FINAL_SCHEMA,
                ),
                dest,
            )
            written[pass_name] += cnt
    check(
        written["p1"] == int(keep["keep_p1"].sum())
        and written["distill"] == int(keep["keep_distill"].sum()),
        f"{setting}: wrote {written} but selected "
        f"{int(keep['keep_p1'].sum())}/{int(keep['keep_distill'].sum())}",
    )
    return written


def step_assemble(cfg, settings):
    from kys3b.pool import load as load_pool

    pool = load_pool(cfg)
    target = cfg.B
    summary = {}
    for setting in settings:
        sdef = cfg.setting(setting)
        mode = sdef["assemble"]
        n = _n_shards(cfg, setting)
        p1_dir, d_dir = _pass_dirs(cfg, setting)
        is_wrap = sdef["rewrite"]["pass1"] == "wrap"
        log(f"=== assemble {setting} (mode={mode}, target={target:,}) ===")

        if mode == "post_rewrite_fasttext":
            # _load_scored `check`s that BOTH passes' scored shards exist, which is the real
            # Decision-3 gate at this point; assert it explicitly so the intent is visible.
            d = cfg.stage("filtered") / setting / "scored_pool"
            rw.require_both_passes(
                sum((d / f"p1_{k:05d}.parquet").exists() for k in range(n)),
                sum((d / f"distill_{k:05d}.parquet").exists() for k in range(n)),
                n,
            )
            scored = _load_scored(cfg, n)
            pool_flat = rw.build_pool(
                dict(doc_id=scored["p1"]["doc_id"], status=scored["p1"]["status"],
                     len=scored["p1"]["len"]),
                dict(doc_id=scored["distill"]["doc_id"], status=scored["distill"]["status"],
                     len=scored["distill"]["len"]),
            )
            s2a = np.flatnonzero(scored["p1"]["status"] == 2)
            s2b = np.flatnonzero(scored["distill"]["status"] == 2)
            raw = np.concatenate([scored["p1"]["score"][s2a], scored["distill"]["score"][s2b]])
            res = rw.filter_top_b(pool_flat, raw, target)
            keep_p1 = np.zeros(scored["p1"]["doc_id"].size, bool)
            keep_d = np.zeros(scored["distill"]["doc_id"].size, bool)
            km = res["kept_mask"]
            keep_p1[pool_flat["row"][km & (pool_flat["source_prompt"] == 0)]] = True
            keep_d[pool_flat["row"][km & (pool_flat["source_prompt"] == 1)]] = True
            keep = dict(keep_p1=keep_p1, keep_distill=keep_d)
            acct = {k: v for k, v in res.items() if k != "kept_mask"}
            acct.update(total_tokens=res["kept_tokens"], shortfall=0)
        else:
            p1 = attach_keys(
                collect_pass(p1_dir, n, with_style=is_wrap), pool, cfg,
                sdef.get("sort_key"), with_topic=(mode == "per_topic"),
            )
            dis = attach_keys(
                collect_pass(d_dir, n), pool, cfg,
                sdef.get("sort_key"), with_topic=(mode == "per_topic"),
            )
            acct_cross = cross_pass_report(p1, dis, pool)
            if mode == "per_topic":
                # topic proportions from the SELECTED SOURCE SET (1.5B behaviour)
                codes, counts = np.unique(p1["topic"], return_counts=True)
                tok_by_topic = {
                    int(c): int(p1["source_tokens"][p1["topic"] == c].sum()) for c in codes
                }
                tot = sum(tok_by_topic.values())
                props = {c: t / tot for c, t in tok_by_topic.items()}
                keep = asm.assemble_per_topic(p1, dis, target, props)
                keep["per_topic_names"] = {
                    pool.vocab[int(c)]: props[int(c)] for c in codes
                }
            else:
                keep = asm.assemble_flat(p1, dis, target, mode, seed=cfg.seed)
            acct = {k: v for k, v in keep.items() if not k.startswith("keep_")}
            acct["cross_pass"] = acct_cross

        written = _write_assembled(cfg, setting, keep, n, styles_for_p1=is_wrap)
        acct["written_rows"] = written
        acct["assemble_mode"] = mode
        acct["sort_key"] = sdef.get("sort_key")
        summary[setting] = acct
        manifest.write(
            cfg.stage("filtered") / setting, "04_filtered",
            dict(setting=setting, target_tokens=target, **acct),
        )
        log(
            f"  {setting}: total {acct['total_tokens']:,} tok "
            f"(target {target:,}, shortfall {acct.get('shortfall', 0):,})"
        )
    atomic_write_json(summary, cfg.stage("reports") / "step2_assemble.json")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 04: strip / score-rewire / assemble")
    ap.add_argument("--step", required=True, choices=["strip", "score-rewire", "assemble"])
    ap.add_argument("--settings", nargs="*", default=list(REWRITE_SETTINGS))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--ref-shards", type=int, default=None,
        help="score-rewire: use only the first N pool shards for the v2 percentile reference "
        "(report runs only; the committed percentile uses all 200)",
    )
    args = ap.parse_args(argv)
    env_offline()
    cfg = Config.load()
    if args.step == "strip":
        step_strip(cfg, args.settings, args.workers)
    elif args.step == "score-rewire":
        step_score_rewire(cfg, sample_ref_shards=args.ref_shards)
    else:
        step_assemble(cfg, args.settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
