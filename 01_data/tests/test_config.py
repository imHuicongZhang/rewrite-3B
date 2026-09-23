import unittest

from helpers import scaled_config

from kys3b.config import Config
from kys3b.io import Stop


class TestConfig(unittest.TestCase):
    def test_real_config_invariants(self):
        c = Config.load()
        self.assertEqual(c.Bs, 2 * c.B, "B_s must be 2*B")
        self.assertEqual(c.budget("rewire_source"), c.kappa * c.Bs, "rewire must be kappa*B_s")
        self.assertEqual(c.budget("da_per_scorer"), c.Bs, "Option C: DA per-scorer budget == B_s")
        self.assertEqual(c.budget("shared_base"), 10_000_000_000)
        self.assertEqual(c.budget("quality_base"), 10_000_000_000)
        self.assertEqual(c.B, 10_000_000_000)
        self.assertEqual(c.Bs, 20_000_000_000)
        self.assertEqual(c.budget("rewire_source"), 40_000_000_000)
        self.assertEqual(c.lam, 0.5)
        self.assertEqual(c.kappa, 2)

    def test_matched_horizons(self):
        t = Config.load().budgets["training"]
        self.assertEqual(t["horizons"], [20_000_000_000, 40_000_000_000, 60_000_000_000])
        self.assertEqual(len(t["seeds"]), 3)

    def test_both_passes_required_for_every_rewriting_arm(self):
        c = Config.load()
        for s in ("quality-first", "wrap-inspired", "rewire-inspired",
                  "diversity-oriented", "disagreement-aware"):
            self.assertTrue(c.setting(s)["rewrite"]["distill"], f"{s} must generate distill")
        self.assertIsNone(c.setting("quality-base")["rewrite"])

    def test_engine_kwargs_are_exactly_the_1p5b_five(self):
        c = Config.load()
        self.assertEqual(
            c.engine_kwargs(),
            dict(tensor_parallel_size=1, dtype="bfloat16",
                 gpu_memory_utilization=0.85, max_model_len=32768),
        )
        self.assertEqual(c.sampling_kwargs(), dict(temperature=0, top_p=1.0, max_tokens=4096))
        self.assertEqual(c.input_drop("p1"), 30720)
        self.assertEqual(c.input_drop("distill"), 28672)

    def test_extra_engine_kwarg_is_rejected(self):
        c = Config.load()
        c.vllm["engine"]["enforce_eager"] = True
        with self.assertRaises(Stop):
            c.engine_kwargs()

    def test_lambda_other_than_half_is_rejected(self):
        c = Config.load()
        c.budgets["lambda_da"] = 1.0
        with self.assertRaises(Stop):
            c.check_invariants()

    def test_da_per_scorer_must_equal_bs(self):
        c = Config.load()
        c.budgets["budgets"]["da_per_scorer"] = 10_000_000_000
        with self.assertRaises(Stop):
            c.check_invariants()

    def test_scaled_config_preserves_ratios(self):
        c = scaled_config()
        self.assertEqual(c.Bs, 2 * c.B)
        self.assertEqual(c.budget("rewire_source"), c.kappa * c.Bs)
        self.assertEqual(c.budget("da_per_scorer"), c.Bs)


if __name__ == "__main__":
    unittest.main()
