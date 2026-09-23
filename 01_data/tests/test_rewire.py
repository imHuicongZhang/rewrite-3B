"""REWIRE: both passes in the pool, fastText on the REWRITTEN text, global top-B."""
import unittest

import numpy as np
from helpers import pass_arrays

from kys3b.io import Stop
from kys3b.post.rewire import build_pool, filter_top_b, require_both_passes


class TestRequireBothPasses(unittest.TestCase):
    def test_both_complete_passes(self):
        require_both_passes(100, 100, 100)

    def test_missing_distill_is_a_hard_stop(self):
        with self.assertRaises(Stop):
            require_both_passes(100, 0, 100)

    def test_missing_pass1_is_a_hard_stop(self):
        with self.assertRaises(Stop):
            require_both_passes(50, 100, 100)


class TestPoolAndFilter(unittest.TestCase):
    def _pool(self, n=1000, seed=9):
        rng = np.random.default_rng(seed)
        ids = np.arange(n)
        st = np.full(n, 2, np.int8)
        st[:10] = 0
        p1 = pass_arrays(ids, st, rng.integers(200, 600, n))
        dis = pass_arrays(ids, st, rng.integers(120, 400, n))
        pool = build_pool(p1, dis)
        return p1, dis, pool

    def test_pool_contains_both_passes_not_deduplicated(self):
        p1, dis, pool = self._pool()
        n2 = int((p1["status"] == 2).sum())
        self.assertEqual(pool["doc_id"].size, 2 * n2, "both passes, one row each")
        self.assertEqual(int((pool["source_prompt"] == 0).sum()), n2)
        self.assertEqual(int((pool["source_prompt"] == 1).sum()), n2)
        # the same doc_id appears twice
        u, c = np.unique(pool["doc_id"], return_counts=True)
        self.assertTrue((c == 2).all())

    def test_only_status2_rows_enter_the_pool(self):
        p1, dis, pool = self._pool()
        dropped = set(p1["doc_id"][p1["status"] != 2].tolist())
        self.assertEqual(len(dropped & set(pool["doc_id"].tolist())), 0)

    def test_filter_keeps_the_highest_scoring_until_the_budget(self):
        p1, dis, pool = self._pool()
        rng = np.random.default_rng(12)
        score = rng.random(pool["doc_id"].size).astype(np.float32)
        target = int(pool["length"].sum() * 0.3)
        res = filter_top_b(pool, score, target)
        self.assertGreaterEqual(res["kept_tokens"], target)
        km = res["kept_mask"]
        self.assertGreater(float(score[km].min()) + 1e-9, float(score[~km].max()) - 1e-9)
        self.assertAlmostEqual(res["fasttext_score_cutoff"], float(score[km].min()), places=6)
        self.assertLess(res["overshoot"], int(pool["length"].max()))

    def test_accounting_splits_by_pass(self):
        p1, dis, pool = self._pool()
        # bias the score so distill wins, as it did at 1.5B (53.1% of kept tokens)
        score = np.where(pool["source_prompt"] == 1, 0.8, 0.2).astype(np.float32)
        score = score + np.random.default_rng(3).random(score.size).astype(np.float32) * 0.1
        res = filter_top_b(pool, score, int(pool["length"].sum() * 0.3))
        self.assertGreater(res["distill_share_pct"], 60.0, "the higher-scoring pass should dominate")
        self.assertEqual(
            res["kept_from_pass1"]["tokens"] + res["kept_from_distill"]["tokens"],
            res["kept_tokens"],
        )
        self.assertLess(res["acceptance_pct_by_token"], 100.0)

    def test_pool_too_small_is_a_hard_stop(self):
        p1, dis, pool = self._pool()
        score = np.random.default_rng(1).random(pool["doc_id"].size).astype(np.float32)
        with self.assertRaises(Stop):
            filter_top_b(pool, score, int(pool["length"].sum()) + 1)

    def test_a_wiki_only_pool_would_change_the_result(self):
        """Documents the reason Decision 3 forbids skipping distill for REWIRE."""
        p1, dis, pool = self._pool()
        score = np.where(pool["source_prompt"] == 1, 0.8, 0.2).astype(np.float32)
        target = int(pool["length"].sum() * 0.3)
        both = filter_top_b(pool, score, target)
        p1_only = build_pool(p1, pass_arrays(np.array([], np.int64), np.array([], np.int8),
                                            np.array([], np.int64)))
        s1 = score[pool["source_prompt"] == 0]
        alone = filter_top_b(p1_only, s1, target)
        self.assertGreater(both["kept_from_distill"]["tokens"], 0)
        self.assertEqual(alone["kept_from_distill"]["tokens"], 0)
        self.assertNotEqual(
            both["kept_from_pass1"]["tokens"], alone["kept_from_pass1"]["tokens"],
            "dropping distill changes which rewrites are retained",
        )


if __name__ == "__main__":
    unittest.main()
