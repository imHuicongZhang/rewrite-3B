"""Calibration must use EXACT source token counts, not a row-fraction proxy.

The failure mode this pins: source shards are deterministic slices of a doc_id-sorted selection,
so the first N shards need not share the full selection's mean document length.  Estimating the
denominator as `source_budget * sampled_rows / total_rows` then biases r by exactly that
difference.  Here document lengths vary ~10x across shards, so the proxy is badly wrong and the
exact sum is right.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
from helpers import *  # noqa: F401,F403  (sys.path setup)

from kys3b.io import Stop, atomic_write_table
from kys3b.shards import shard_path

from kys3b import calibration as calib


class TestExactCalibration(unittest.TestCase):
    ROWS = 100
    N_SHARDS = 4
    # strongly varying mean document length across shards
    LENS = [50, 150, 300, 500]

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.src = Path(self._d.name) / "src"
        self.out = Path(self._d.name) / "out"
        self.src.mkdir(parents=True)
        self.out.mkdir(parents=True)
        self.truth = []
        for k in range(self.N_SHARDS):
            ids = np.arange(k * self.ROWS, (k + 1) * self.ROWS, dtype=np.int64)
            ntok = np.full(self.ROWS, self.LENS[k], dtype=np.int32)
            atomic_write_table(
                pa.table({
                    "doc_id": pa.array(ids, type=pa.int64()),
                    "text": pa.array(["x"] * self.ROWS, type=pa.large_string()),
                    "tokens_llama2": pa.array(ntok, type=pa.int32()),
                }),
                shard_path(self.src, k),
            )
            status = np.full(self.ROWS, 2, np.int8)
            status[:10] = 0                      # 10% dropped -> census vs status2 differ
            rew = np.where(status == 2, (ntok // 2).astype(np.int64), 0)
            atomic_write_table(
                pa.table({
                    "doc_id": pa.array(ids, type=pa.int64()),
                    "status": pa.array(status, type=pa.int8()),
                    "rewritten_tokens": pa.array(rew.astype(np.int32), type=pa.int32()),
                }),
                shard_path(self.out, k),
            )
            self.truth.append(
                dict(
                    src_all=int((ntok.astype(np.int64) + 1).sum()),
                    src_s2=int((ntok.astype(np.int64) + 1)[status == 2].sum()),
                    out=int(rew[status == 2].sum()),
                )
            )

    def tearDown(self):
        self._d.cleanup()

    def test_per_shard_accounting_is_exact(self):
        for k in range(self.N_SHARDS):
            r = calib.shard_ratio(self.src, self.out, k)
            self.assertEqual(r["src_train_tokens_all"], self.truth[k]["src_all"])
            self.assertEqual(r["src_train_tokens_status2"], self.truth[k]["src_s2"])
            self.assertEqual(r["out_tokens_status2"], self.truth[k]["out"])
            self.assertEqual(r["rows"], self.ROWS)

    def test_sampling_the_short_shards_is_not_biased_by_a_row_fraction_proxy(self):
        """Sample the 2 SHORTEST shards; the proxy over-states the denominator badly."""
        sampled = [0, 1]
        exact_src = sum(self.truth[k]["src_all"] for k in sampled)
        exact_out = sum(self.truth[k]["out"] for k in sampled)
        r_exact = exact_out / exact_src

        # what the OLD implementation would have computed
        full_budget = sum(t["src_all"] for t in self.truth)
        rows_frac = (len(sampled) * self.ROWS) / (self.N_SHARDS * self.ROWS)
        r_proxy = exact_out / (full_budget * rows_frac)

        got = dict(src_train_tokens_all=0, out_tokens_status2=0)
        for k in sampled:
            r = calib.shard_ratio(self.src, self.out, k)
            got["src_train_tokens_all"] += r["src_train_tokens_all"]
            got["out_tokens_status2"] += r["out_tokens_status2"]
        r_measured = got["out_tokens_status2"] / got["src_train_tokens_all"]

        self.assertAlmostEqual(r_measured, r_exact, places=12, msg="calibration must be exact")
        # the proxy is wrong by >50% here -- exactly the bias the fix removes
        self.assertGreater(
            abs(r_proxy - r_exact) / r_exact, 0.5,
            "the test data must make the row-fraction proxy clearly wrong",
        )

    def test_census_and_status2_ratios_differ_by_the_dropped_share(self):
        r = calib.shard_ratio(self.src, self.out, 0)
        r_census = r["out_tokens_status2"] / r["src_train_tokens_all"]
        r_s2 = r["out_tokens_status2"] / r["src_train_tokens_status2"]
        self.assertGreater(r_s2, r_census, "status2-only compression is the larger ratio")
        self.assertAlmostEqual(r_census / r_s2, 0.9, places=6, msg="10% of rows were dropped")

    def test_misaligned_rows_are_a_hard_stop(self):
        ids = np.arange(999, 999 + self.ROWS, dtype=np.int64)
        atomic_write_table(
            pa.table({
                "doc_id": pa.array(ids, type=pa.int64()),
                "status": pa.array(np.full(self.ROWS, 2, np.int8), type=pa.int8()),
                "rewritten_tokens": pa.array(np.ones(self.ROWS, np.int32), type=pa.int32()),
            }),
            shard_path(self.out, 0),
        )
        with self.assertRaises(Stop):
            calib.shard_ratio(self.src, self.out, 0)

    def test_reference_ratios_are_the_1p5b_census_values(self):
        self.assertEqual(calib.REF_R["quality-first"], dict(p1=0.3399, distill=0.2581))
        self.assertEqual(calib.REF_R["rewire-inspired"], dict(p1=0.4628, distill=0.3660))
        self.assertEqual(len(calib.REF_R), 5)


if __name__ == "__main__":
    unittest.main()
