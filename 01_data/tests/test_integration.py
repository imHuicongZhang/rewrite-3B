"""End-to-end integration on a synthetic mini-pool written to real parquet files.

Exercises the production code paths -- pool.load, select_all, shards.materialize,
strip.strip_shard, collect_pass, assemble_*, rewire.filter_top_b, mix + bucketed_shuffle --
at 1/500,000 of the production budgets, with the budget RATIOS preserved exactly.

The only thing faked is the GPU: a stand-in writes plausible rewritten shards (including
artifact preambles, status 0/1/2 rows and wrap styles) in the exact schema the real worker
emits.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from helpers import TOPICS, scaled_config

from kys3b.config import Config  # used in the mini_config return annotation
from kys3b.io import atomic_write_table
from kys3b.post import assemble as asm
from kys3b.post import rewire as rwm
from kys3b.post.collect import attach_keys, collect_pass, cross_pass_report
from kys3b.post.mix import base_table, check_no_overlap, horizon_plan
from kys3b.pool import load as load_pool
from kys3b.post.strip import WIKI_PREFIX, strip_shard
from kys3b.prompts import assign_wrap_styles
from kys3b.select import select_all
from kys3b.shards import load_index, materialize, shard_path

N_SHARDS, ROWS_FULL, ROWS_LAST = 4, 500, 300
N_DOCS = (N_SHARDS - 1) * ROWS_FULL + ROWS_LAST
SCALE = 500_000
ROWS_PER_SOURCE_SHARD = 100


class FakeLlama2:
    """Whitespace tokenizer with the transformers call signature (count_batch's contract)."""

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [t.split() for t in texts]}


def write_mini_pool(root: Path, seed=5) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    did = 0
    for i in range(N_SHARDS):
        n = ROWS_FULL if i < N_SHARDS - 1 else ROWS_LAST
        ids = np.arange(did, did + n, dtype=np.int64)
        did += n
        ft = rng.random(n).astype(np.float32)
        ft[ft < 0.09] = np.float32(0.0433)            # the tie floor
        ft = np.clip(ft, 1e-6, 1.0).astype(np.float32)
        fw = np.clip(0.45 * ft + 0.55 * rng.random(n), 1e-6, 1.0).astype(np.float32)
        mb = np.clip(0.40 * ft + 0.60 * rng.random(n), 1e-6, 1.0).astype(np.float32)
        ntok = rng.integers(40, 400, n).astype(np.int32)
        topics = [TOPICS[k] for k in rng.integers(0, 24, n)]
        texts = [" ".join(f"w{j}" for j in range(int(t))) for t in ntok]
        atomic_write_table(
            pa.table(
                {
                    "orig_doc_id": pa.array(ids, type=pa.int64()),
                    "doc_id": pa.array(ids, type=pa.int64()),
                    "text": pa.array(texts, type=pa.large_string()),
                    "url": pa.array([f"http://e/{x}" for x in ids], type=pa.large_string()),
                    "metadata": pa.array(["{}"] * n, type=pa.large_string()),
                    "fasttext": pa.array(ft, type=pa.float32()),
                    "fineweb-edu": pa.array(fw, type=pa.float32()),
                    "modernbert": pa.array(mb, type=pa.float32()),
                    "topic": pa.array(topics, type=pa.large_string()),
                    "fasttext-ranking-v2": pa.array(ft, type=pa.float32()),
                    "fineweb-edu-ranking-v2": pa.array(fw, type=pa.float32()),
                    "modernbert-ranking-v2": pa.array(mb, type=pa.float32()),
                    "tokens-llama2": pa.array(ntok, type=pa.int32()),
                }
            ),
            root / f"merged_clean_{i:05d}.parquet",
        )


def mini_config(tmp: Path) -> Config:
    cfg = scaled_config(scale=SCALE, tmp_root=tmp)
    cfg.pool["n_shards"] = N_SHARDS
    cfg.pool["rows_full"] = ROWS_FULL
    cfg.pool["rows_last"] = ROWS_LAST
    cfg.pool["n_docs"] = N_DOCS
    cfg.pool["val_size"] = 20
    cfg.budgets["sharding"]["rows_per_shard"] = ROWS_PER_SOURCE_SHARD
    cfg.check_invariants()
    return cfg


def fake_rewrite(cfg, setting, pass_name, rng):
    """Write rewritten shards in the exact schema the real worker emits."""
    sdef = cfg.setting(setting)
    is_wrap = pass_name == "p1" and sdef["rewrite"]["pass1"] == "wrap"
    src = cfg.stage("sources") / setting
    out = cfg.stage("rewritten") / setting / pass_name
    out.mkdir(parents=True, exist_ok=True)
    idx = load_index(src)
    for k in range(idx["n_shards"]):
        t = pq.read_table(shard_path(src, k), use_threads=False)
        ids = t.column("doc_id").to_numpy(zero_copy_only=False)
        texts = t.column("text").to_pylist()
        n = len(texts)
        styles = assign_wrap_styles(k, n, cfg.seed) if is_wrap else [None] * n
        status, rewritten, finish, ntok, nin = [], [], [], [], []
        for j, tx in enumerate(texts):
            words = tx.split()
            # 2% dropped over-length, 1% truncated -- the 1.5B status mix, roughly
            r = rng.random()
            # the MEASURED 1.5B compression ratios (DESIGN_DELTA section 5), so pass 1 alone
            # falls short of B and the distill top-up path is genuinely exercised
            keep = max(1, int(len(words) * (0.34 if pass_name == "p1" else 0.26)))
            body = " ".join(words[:keep])
            if r < 0.02:
                status.append(0); rewritten.append(""); finish.append("")
            elif r < 0.03:
                status.append(1); rewritten.append(body); finish.append("length")
            else:
                if pass_name == "p1" and not is_wrap and rng.random() < 0.9:
                    body = WIKI_PREFIX + body      # the artifact the strip step removes
                elif pass_name == "distill" and rng.random() < 0.05:
                    body = "### Paraphrased Text\n\n" + body
                status.append(2); rewritten.append(body); finish.append("stop")
            ntok.append(len(rewritten[-1].split()))
            nin.append(len(words) + 150)
        cols = {
            "doc_id": pa.array(ids, type=pa.int64()),
            "rewritten": pa.array(rewritten, type=pa.large_string()),
            "rewritten_tokens": pa.array(np.array(ntok, np.int32), type=pa.int32()),
            "status": pa.array(np.array(status, np.int8), type=pa.int8()),
            "finish_reason": pa.array(finish, type=pa.large_string()),
            "input_tokens_qwen": pa.array(np.array(nin, np.int32), type=pa.int32()),
        }
        if is_wrap:
            cols["wrap_style"] = pa.array(styles, type=pa.large_string())
        atomic_write_table(pa.table(cols), shard_path(out, k))


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.cfg = mini_config(cls.tmp)
        write_mini_pool(cls.cfg.root("pool"))
        cls.pool = load_pool(cls.cfg, workers=2)
        cls.sel = select_all(cls.pool, cls.cfg)
        cls.ltok = FakeLlama2()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ---------------------------------------------------------------- stage 00/01
    def test_00_pool_loads_with_the_published_conventions(self):
        from kys3b.pool import validate_files

        info = validate_files(self.cfg)
        self.assertEqual(info["total_rows"], N_DOCS)
        self.assertEqual(len(self.pool.vocab), 24)
        # TRAIN length is tokens-llama2 + 1
        t = pq.read_table(self.cfg.pool_shard(0), columns=["tokens-llama2"], use_threads=False)
        raw = t.column("tokens-llama2").to_numpy(zero_copy_only=False).astype(np.int64)
        self.assertTrue(np.array_equal(self.pool.tok[: raw.size], raw + 1))

    def test_01_selection_invariants_hold_on_real_files(self):
        base = set(self.sel.base.idx.tolist())
        for name, blk in self.sel.blocks.items():
            self.assertEqual(len(base & set(blk.idx.tolist())), 0, name)
        self.assertEqual(
            self.sel.blocks["rewire-inspired"].target, 2 * self.cfg.Bs, "kappa = 2"
        )
        self.assertEqual(
            self.sel.blocks["disagreement-aware"].extra["per_scorer_budget"], self.cfg.Bs
        )

    # ---------------------------------------------------------------- stage 02
    def test_02_materialize_is_idempotent_and_fingerprinted(self):
        setting = "quality-first"
        ids = np.sort(self.sel.blocks[setting].idx)
        out = self.cfg.stage("sources") / setting
        a = materialize(self.cfg, setting, ids, out, ROWS_PER_SOURCE_SHARD)
        mtimes = {p.name: p.stat().st_mtime_ns for p in sorted(out.glob("shard_*.parquet"))}
        b = materialize(self.cfg, setting, ids, out, ROWS_PER_SOURCE_SHARD)
        self.assertEqual(a["n_shards"], b["n_shards"])
        again = {p.name: p.stat().st_mtime_ns for p in sorted(out.glob("shard_*.parquet"))}
        self.assertEqual(mtimes, again, "a rerun must not rewrite completed shards")
        # every selected doc_id appears exactly once, with the pool's own text
        seen = np.concatenate(
            [
                pq.read_table(p, columns=["doc_id"]).column("doc_id").to_numpy(
                    zero_copy_only=False
                )
                for p in sorted(out.glob("shard_*.parquet"))
            ]
        )
        self.assertTrue(np.array_equal(np.sort(seen), ids))

    def test_02_materialize_output_is_independent_of_the_batch_size(self):
        """Peak memory is bounded by `batch_shards`; the output must not depend on it.

        At production scale quality-first is ~12.2M documents over ~1,220 shards, so buffering
        every shard's text at once would be ~80 GB.  Batching is the fix, and this pins that it
        changes nothing observable.
        """
        ids = np.arange(0, N_DOCS, 3, dtype=np.int64)
        out = self.cfg.stage("sources") / "_batchprobe"
        results = []
        for bs in (1, 3, 1000):
            for f in out.glob("shard_*.parquet"):
                f.unlink()
            idx = materialize(self.cfg, "_batchprobe", ids, out, 100, batch_shards=bs)
            got = np.concatenate([
                pq.read_table(out / s["file"], columns=["doc_id"])
                .column("doc_id").to_numpy(zero_copy_only=False)
                for s in idx["shards"]
            ])
            results.append((idx["n_shards"], got))
        self.assertEqual(len({r[0] for r in results}), 1, "shard count must not depend on batching")
        for _, got in results:
            self.assertTrue(np.array_equal(got, ids))

    # ---------------------------------------------------------------- stages 03-05
    def _run_arm(self, setting):
        ids = np.sort(self.sel.blocks[setting].idx)
        src = self.cfg.stage("sources") / setting
        materialize(self.cfg, setting, ids, src, ROWS_PER_SOURCE_SHARD)
        rng = np.random.default_rng(hash(setting) % 2**31)
        for pass_name in ("p1", "distill"):
            fake_rewrite(self.cfg, setting, pass_name, rng)
        n = load_index(src)["n_shards"]
        # strip is in-place and idempotent
        for pass_name in ("p1", "distill"):
            d = self.cfg.stage("rewritten") / setting / pass_name
            first = [strip_shard(shard_path(d, k), setting, pass_name, self.ltok) for k in range(n)]
            second = [strip_shard(shard_path(d, k), setting, pass_name, self.ltok) for k in range(n)]
            self.assertEqual(
                sum(x["n_stripped"] for x in second), 0,
                f"{setting}/{pass_name}: strip must be idempotent",
            )
            if pass_name == "p1" and setting != "wrap-inspired":
                self.assertGreater(
                    sum(x["n_stripped"] for x in first), 0, "the wiki prefix should be found"
                )
        return n

    def test_03_all_five_arms_round_trip(self):
        cfg = self.cfg
        results = {}
        self._distill_was_needed = False
        for setting in (
            "quality-first", "wrap-inspired", "diversity-oriented",
            "disagreement-aware", "rewire-inspired",
        ):
            n = self._run_arm(setting)
            sdef = cfg.setting(setting)
            mode = sdef["assemble"]
            p1_dir = cfg.stage("rewritten") / setting / "p1"
            d_dir = cfg.stage("rewritten") / setting / "distill"
            is_wrap = sdef["rewrite"]["pass1"] == "wrap"

            if mode == "post_rewrite_fasttext":
                p1 = collect_pass(p1_dir, n)
                dis = collect_pass(d_dir, n)
                rwm.require_both_passes(n, n, n)
                flat = rwm.build_pool(p1, dis)
                rng = np.random.default_rng(1)
                score = rng.random(flat["doc_id"].size).astype(np.float32)
                res = rwm.filter_top_b(flat, score, cfg.B)
                self.assertGreaterEqual(res["kept_tokens"], cfg.B)
                self.assertGreater(res["kept_from_pass1"]["docs"], 0)
                self.assertGreater(res["kept_from_distill"]["docs"], 0)
                results[setting] = res["kept_tokens"]
                continue

            p1 = attach_keys(
                collect_pass(p1_dir, n, with_style=is_wrap), self.pool, cfg,
                sdef.get("sort_key"), with_topic=(mode == "per_topic"),
            )
            dis = attach_keys(
                collect_pass(d_dir, n), self.pool, cfg, sdef.get("sort_key"),
                with_topic=(mode == "per_topic"),
            )
            cross = cross_pass_report(p1, dis, self.pool)
            self.assertEqual(cross["paired_docs"], p1["doc_id"].size)
            self.assertGreater(cross["source_coverage_pct"], 90.0)

            if mode == "per_topic":
                codes = np.unique(p1["topic"])
                tokb = {int(c): int(p1["source_tokens"][p1["topic"] == c].sum()) for c in codes}
                tot = sum(tokb.values())
                keep = asm.assemble_per_topic(p1, dis, cfg.B, {c: v / tot for c, v in tokb.items()})
                # policy A: no topic may exceed its quota by more than one document
                for row in keep["per_topic"]:
                    self.assertLessEqual(
                        row["tokens"], row["quota_tokens"] + 2000,
                        f"topic {row['topic']} looks back-filled",
                    )
            else:
                keep = asm.assemble_flat(p1, dis, cfg.B, mode, seed=cfg.seed)
                if keep["used_distill"]:
                    self._distill_was_needed = True

            # only status==2 rows may ever be kept
            self.assertFalse(keep["keep_p1"][p1["status"] != 2].any())
            self.assertFalse(keep["keep_distill"][dis["status"] != 2].any())
            results[setting] = keep["total_tokens"]

        self.assertEqual(len(results), 5)
        for setting, tok in results.items():
            if setting == "diversity-oriented":
                self.assertGreater(tok, 0.8 * self.cfg.B, f"{setting} fell too far short")
            else:
                self.assertGreaterEqual(tok, self.cfg.B, setting)
        # with the measured 1.5B ratios, pass 1 alone must NOT fill B -- the distill pass is
        # required, which is exactly the finding behind Decision 3
        self.assertTrue(self._distill_was_needed, "distill top-up path was not exercised")

    def test_04_final_mix_shuffle_and_matched_horizons(self):
        cfg = self.cfg
        base_ids = np.sort(self.sel.base.idx)
        fin = cfg.stage("final") / "quality-base"
        base_dir, strat_dir = fin / "shared-base-10B", fin / "strategy"
        qb_ids = np.sort(self.sel.blocks["quality-base"].idx)

        for ids, d in ((base_ids, base_dir), (qb_ids, strat_dir)):
            d.mkdir(parents=True, exist_ok=True)
            texts, toks = {}, {}
            for i in range(N_SHARDS):
                t = pq.read_table(
                    cfg.pool_shard(i), columns=["doc_id", "text", "tokens-llama2"],
                    use_threads=False,
                )
                pid = t.column("doc_id").to_numpy(zero_copy_only=False)
                tx = t.column("text").to_pylist()
                tk = t.column("tokens-llama2").to_numpy(zero_copy_only=False)
                keep = np.isin(pid, ids)
                for j in np.flatnonzero(keep):
                    texts[int(pid[j])] = tx[int(j)]
                    toks[int(pid[j])] = int(tk[int(j)])
            atomic_write_table(
                base_table(ids, [texts[int(x)] for x in ids], [toks[int(x)] for x in ids]),
                d / "part_00000.parquet",
            )

        check_no_overlap(base_ids, qb_ids)

        from kys3b.post.mix import shuffle_final

        res = shuffle_final(
            [base_dir / "part_00000.parquet", strat_dir / "part_00000.parquet"],
            fin / "shuffled", fin / "_tmp", seed=cfg.seed, rows_per_out_shard=400, mem_gb=2.0,
        )
        self.assertEqual(res["rows"], base_ids.size + qb_ids.size)
        got = np.concatenate(
            [
                pq.read_table(p, columns=["orig_doc_id"]).column("orig_doc_id").to_numpy(
                    zero_copy_only=False
                )
                for p in sorted((fin / "shuffled").glob("part_*.parquet"))
            ]
        )
        self.assertTrue(
            np.array_equal(np.sort(got), np.sort(np.concatenate([base_ids, qb_ids]))),
            "the shuffle must conserve exactly the documents that went in",
        )

        # the training handoff: matched TOKEN horizons, never corpus_size * n_epochs
        ep = int(cfg.budgets["training"]["epoch_tokens"])
        plan_full = horizon_plan(ep, cfg)
        self.assertEqual(plan_full["horizons"], [ep, 2 * ep, 3 * ep])
        self.assertEqual(plan_full["passes_to_horizon"], [1.0, 2.0, 3.0])
        short = horizon_plan(int(ep * 0.989), cfg)
        self.assertEqual(short["horizons"], [ep, 2 * ep, 3 * ep], "horizons must NOT shrink")
        self.assertGreater(short["passes_to_horizon"][2], 3.0, "a short corpus is re-read more")
        self.assertLess(short["passes_to_horizon"][2], 3.1)


if __name__ == "__main__":
    unittest.main()
