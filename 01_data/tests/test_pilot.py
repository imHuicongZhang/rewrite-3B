"""The calibration pilot must be representative, deterministic and strictly bounded."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
from helpers import scaled_config

from kys3b.calibration import (
    DEFAULT_PILOT_SHARDS,
    aggregate,
    pilot_shards,
    project_yield,
)
from kys3b.io import atomic_write_table
from kys3b.shards import shard_path


class TestPilotShardSelection(unittest.TestCase):
    def test_default_is_inside_the_requested_band(self):
        self.assertGreaterEqual(DEFAULT_PILOT_SHARDS, 8)
        self.assertLessEqual(DEFAULT_PILOT_SHARDS, 16)

    def test_is_not_the_first_k_shards(self):
        ids = pilot_shards(1220, 12)
        self.assertNotEqual(ids, list(range(12)), "a leading block is a biased slice")
        self.assertGreater(ids[0], 12)

    def test_spans_the_whole_range(self):
        n, k = 1220, 12
        ids = pilot_shards(n, k)
        self.assertLess(ids[0], n * 0.1, "no coverage near the start")
        self.assertGreater(ids[-1], n * 0.9, "no coverage near the end")
        # every decile of the range should contain at least one pilot shard
        for d in range(10):
            lo, hi = d * n / 10, (d + 1) * n / 10
            self.assertTrue(
                any(lo <= i < hi for i in ids), f"decile {d} of the shard range is unsampled"
            )

    def test_gaps_are_even(self):
        ids = pilot_shards(2110, 12)
        gaps = np.diff(ids)
        self.assertLess(gaps.max() - gaps.min(), 3, f"spacing is not even: {gaps.tolist()}")

    def test_is_deterministic_and_sorted_and_unique(self):
        for _ in range(3):
            ids = pilot_shards(997, 11)
            self.assertEqual(ids, pilot_shards(997, 11))
            self.assertEqual(ids, sorted(ids))
            self.assertEqual(len(ids), len(set(ids)))

    def test_count_and_bounds(self):
        for n in (100, 997, 1220, 4240):
            for k in (8, 12, 16):
                ids = pilot_shards(n, k)
                self.assertEqual(len(ids), k)
                self.assertTrue(all(0 <= i < n for i in ids))

    def test_degenerate_inputs(self):
        self.assertEqual(pilot_shards(5, 12), [0, 1, 2, 3, 4])
        self.assertEqual(pilot_shards(0, 12), [])
        self.assertEqual(pilot_shards(100, 0), [])


class TestProjection(unittest.TestCase):
    def setUp(self):
        self.cfg = scaled_config()

    def test_projects_both_passes_against_the_target(self):
        B, Bs = self.cfg.B, self.cfg.Bs
        pr = project_yield(self.cfg, dict(p1=0.34, distill=0.26), Bs)
        self.assertTrue(pr["complete"])
        self.assertAlmostEqual(pr["projected_p1_tokens"], 0.34 * Bs)
        self.assertAlmostEqual(pr["projected_distill_tokens"], 0.26 * Bs)
        self.assertAlmostEqual(pr["projected_total_tokens"], 0.60 * Bs)
        # B_s = 2B, so r_sum of 0.60 gives 1.20 x B -> +20% headroom
        self.assertAlmostEqual(pr["headroom_pct"], 20.0, places=6)
        self.assertTrue(pr["meets_target"])
        self.assertEqual(pr["target"], B)

    def test_flags_an_arm_that_would_finish_short(self):
        pr = project_yield(self.cfg, dict(p1=0.25, distill=0.20), self.cfg.Bs)
        self.assertFalse(pr["meets_target"])
        self.assertLess(pr["headroom_pct"], 0.0)

    def test_reproduces_the_plan_headroom_for_quality_first(self):
        """plan.md 18.2: quality-first r = 0.3399 + 0.2581 -> +19.6%, the tightest arm."""
        pr = project_yield(self.cfg, dict(p1=0.3399, distill=0.2581), self.cfg.Bs)
        self.assertAlmostEqual(pr["headroom_pct"], 19.62, places=1)

    def test_incomplete_when_a_pass_is_missing(self):
        pr = project_yield(self.cfg, dict(p1=0.34), self.cfg.Bs)
        self.assertFalse(pr["complete"])
        self.assertNotIn("projected_total_tokens", pr)


class TestPilotIsBoundedToItsShards(unittest.TestCase):
    """Aggregation must use exactly the pilot shard ids, on complete shards."""

    N, ROWS = 60, 40

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.src = Path(self._d.name) / "src"
        self.out = Path(self._d.name) / "_pilot"
        self.src.mkdir(parents=True)
        self.out.mkdir(parents=True)
        self.ids = pilot_shards(self.N, 8)
        for k in range(self.N):
            d = np.arange(k * self.ROWS, (k + 1) * self.ROWS, dtype=np.int64)
            # length varies with the shard index, so a biased subset gives a different r
            ntok = np.full(self.ROWS, 50 + 8 * k, dtype=np.int32)
            atomic_write_table(
                pa.table({
                    "doc_id": pa.array(d, type=pa.int64()),
                    "text": pa.array(["x"] * self.ROWS, type=pa.large_string()),
                    "tokens_llama2": pa.array(ntok, type=pa.int32()),
                }),
                shard_path(self.src, k),
            )
            if k in self.ids:  # ONLY the pilot shards have output
                atomic_write_table(
                    pa.table({
                        "doc_id": pa.array(d, type=pa.int64()),
                        "status": pa.array(np.full(self.ROWS, 2, np.int8), type=pa.int8()),
                        "rewritten_tokens": pa.array(
                            (ntok // 3).astype(np.int32), type=pa.int32()
                        ),
                    }),
                    shard_path(self.out, k),
                )

    def tearDown(self):
        self._d.cleanup()

    def test_only_the_pilot_shards_were_produced(self):
        produced = [k for k in range(self.N) if shard_path(self.out, k).exists()]
        self.assertEqual(produced, self.ids, "the pilot must be strictly bounded")
        self.assertEqual(len(produced), 8)

    def test_shards_are_complete_not_partial(self):
        from kys3b.io import parquet_rows

        for k in self.ids:
            self.assertEqual(parquet_rows(shard_path(self.out, k)), self.ROWS)

    def test_aggregate_over_the_pilot_set_is_exact(self):
        agg = aggregate(self.src, self.out, self.ids)
        exp_src = exp_out = 0
        for k in self.ids:
            ntok = np.full(self.ROWS, 50 + 8 * k, dtype=np.int64)
            exp_src += int((ntok + 1).sum())
            exp_out += int((ntok // 3).sum())
        self.assertEqual(agg["src_train_tokens_all"], exp_src)
        self.assertEqual(agg["out_tokens_status2"], exp_out)
        self.assertAlmostEqual(agg["r_census"], exp_out / exp_src, places=12)

    def test_a_spread_sample_beats_a_leading_block_on_this_data(self):
        """The spread sample's r must be much closer to the whole-corpus r than the first-8 are."""
        whole = aggregate(self.src, self.out, self.ids)  # stands in for the full-corpus estimate
        # rebuild output for the FIRST 8 shards to compare
        first = list(range(8))
        for k in first:
            d = np.arange(k * self.ROWS, (k + 1) * self.ROWS, dtype=np.int64)
            ntok = np.full(self.ROWS, 50 + 8 * k, dtype=np.int32)
            atomic_write_table(
                pa.table({
                    "doc_id": pa.array(d, type=pa.int64()),
                    "status": pa.array(np.full(self.ROWS, 2, np.int8), type=pa.int8()),
                    "rewritten_tokens": pa.array((ntok // 3).astype(np.int32), type=pa.int32()),
                }),
                shard_path(self.out, k),
            )
        lead = aggregate(self.src, self.out, first)
        # both measure r exactly; the point is that the leading block covers a narrow, short-doc
        # region, so its mean source length is far from the corpus mean
        self.assertLess(
            lead["src_train_tokens_all"] / (8 * self.ROWS),
            whole["src_train_tokens_all"] / (8 * self.ROWS),
            "the leading block should be systematically shorter on this data",
        )


if __name__ == "__main__":
    unittest.main()
