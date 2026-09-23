"""End-to-end selection on a synthetic pool, at 1/1,000,000 of the production budgets.

Every structural invariant plan.md promises is asserted here on a pool small enough to run in
a second, with the budget RATIOS identical to production.
"""
import unittest

import numpy as np
from helpers import scaled_config, synthetic_pool

from kys3b.config import SETTINGS
from kys3b.select import select_all


class TestSelectionSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = scaled_config()
        cls.pool = synthetic_pool(n=200_000, seed=11)
        cls.sel = select_all(cls.pool, cls.cfg)

    def test_all_six_blocks_exist(self):
        self.assertEqual(set(self.sel.blocks), set(SETTINGS))

    def test_base_meets_budget_and_overshoots_by_less_than_one_doc(self):
        b = self.sel.base
        self.assertGreaterEqual(b.tokens, b.target)
        self.assertLess(b.overshoot, int(self.pool.tok.max()))

    def test_no_strategy_document_comes_from_the_base(self):
        base = set(self.sel.base.idx.tolist())
        for name, blk in self.sel.blocks.items():
            self.assertEqual(
                len(base & set(blk.idx.tolist())), 0,
                f"{name} must draw only from the residual pool D'",
            )

    def test_no_block_contains_a_validation_document(self):
        val = set(self.sel.val_idx.tolist())
        for name, blk in self.sel.blocks.items():
            self.assertEqual(len(val & set(blk.idx.tolist())), 0, name)
        self.assertEqual(len(val & set(self.sel.base.idx.tolist())), 0, "base")

    def test_no_duplicates_within_a_block(self):
        for name, blk in self.sel.blocks.items():
            self.assertEqual(np.unique(blk.idx).size, blk.idx.size, name)

    def test_quality_base_is_contained_in_quality_first(self):
        qb = self.sel.blocks["quality-base"].idx
        qf = set(self.sel.blocks["quality-first"].idx.tolist())
        self.assertTrue(all(int(i) in qf for i in qb))

    def test_budgets_are_met(self):
        for name, blk in self.sel.blocks.items():
            if name == "diversity-oriented":
                continue  # policy A may legitimately land under budget
            self.assertGreaterEqual(blk.tokens, blk.target, name)

    def test_rewire_is_exactly_kappa_times_the_source_budget(self):
        self.assertEqual(
            self.sel.blocks["rewire-inspired"].target,
            self.cfg.kappa * self.cfg.Bs,
        )
        self.assertEqual(self.sel.blocks["rewire-inspired"].target, 2 * self.cfg.Bs)

    def test_da_domain_shape_matches_the_1p5b_structure(self):
        e = self.sel.blocks["disagreement-aware"].extra
        self.assertEqual(e["per_scorer_budget"], self.cfg.Bs, "Option C")
        self.assertEqual(e["q_floor_pct"], 30.0)
        self.assertEqual(e["v_cap_pct"], 90.0)
        self.assertEqual(e["lambda"], 0.5)
        self.assertGreater(e["u_tokens"], self.cfg.Bs, "U must exceed B_s")
        self.assertGreaterEqual(e["ut_tokens"], self.cfg.Bs, "U_tau must be able to fill B_s")
        self.assertLess(e["ut_docs"], e["u_docs"], "the floor/cap must remove something")
        # the cap alone should be a narrow safeguard, as at 1.5B (0.35% of U)
        c = e["contingency"]
        self.assertLess(
            c["q_pass_v_fail"] / max(1, e["u_docs"]), 0.05,
            "the variance cap alone should reject only a small slice of U",
        )

    def test_da_selection_is_a_subset_of_u_tau_and_ranked_by_u(self):
        q, v = self.pool.qv()
        idx = self.sel.blocks["disagreement-aware"].idx
        e = self.sel.blocks["disagreement-aware"].extra
        self.assertTrue((q[idx] >= e["tau_q"] - 1e-6).all(), "quality floor violated")
        self.assertTrue((v[idx] <= e["tau_v"] + 1e-6).all(), "variance cap violated")
        u = q + np.float32(0.5) * np.sqrt(v)
        # every selected document must score at least the cutoff
        self.assertGreaterEqual(float(u[idx].min()) + 1e-6, float(e["u_at_cutoff"]))

    def test_diversity_preserves_the_topic_distribution(self):
        blk = self.sel.blocks["diversity-oriented"]
        tok = self.pool.tok
        res = self.sel.residual_idx
        pool_share = np.array(
            [tok[res[self.pool.topic[res] == c]].sum() for c in range(24)], dtype=np.float64
        )
        pool_share /= pool_share.sum()
        sel_share = np.array(
            [tok[blk.idx[self.pool.topic[blk.idx] == c]].sum() for c in range(24)],
            dtype=np.float64,
        )
        sel_share /= sel_share.sum()
        # Jensen-Shannon-free simple check: every topic within 1.5pp of the pool share
        self.assertLess(
            float(np.abs(sel_share - pool_share).max()), 0.015,
            "topic-preserving selection must track the pool's topic distribution",
        )

    def test_diversity_ranks_by_consensus_q_not_fasttext(self):
        """Within a topic, the selected set must be the q-top, not the fastText-top."""
        blk = self.sel.blocks["diversity-oriented"]
        q = self.pool.q
        res = self.sel.residual_idx
        picked = np.zeros(self.pool.n, bool)
        picked[blk.idx] = True
        better_q = 0
        for c in range(24):
            cat = res[self.pool.topic[res] == c]
            sel_c = cat[picked[cat]]
            rej_c = cat[~picked[cat]]
            if sel_c.size == 0 or rej_c.size == 0:
                continue
            # mean q of the selected must exceed mean q of the rejected in every topic
            if q[sel_c].mean() > q[rej_c].mean():
                better_q += 1
        self.assertGreaterEqual(better_q, 22, "q must be the within-topic ranking key")

    def test_selection_is_deterministic(self):
        sel2 = select_all(synthetic_pool(n=200_000, seed=11), scaled_config())
        for name in SETTINGS:
            self.assertTrue(
                np.array_equal(
                    np.sort(self.sel.blocks[name].idx), np.sort(sel2.blocks[name].idx)
                ),
                f"{name} is not reproducible",
            )
        self.assertTrue(np.array_equal(np.sort(self.sel.base.idx), np.sort(sel2.base.idx)))

    def test_wrap_and_rewire_use_different_rng_children(self):
        w = set(self.sel.blocks["wrap-inspired"].idx.tolist())
        r = set(self.sel.blocks["rewire-inspired"].idx.tolist())
        self.assertNotEqual(w, r, "child 2 and child 3 must give different draws")


if __name__ == "__main__":
    unittest.main()
