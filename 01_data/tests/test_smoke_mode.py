"""Bounded smoke mode must be genuinely bounded and must never corrupt production bookkeeping."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa

from helpers import *  # noqa: F401,F403  (sys.path setup)

from kys3b.claims import ClaimDir
from kys3b.io import atomic_write_table, parquet_rows
from kys3b.prompts import assign_wrap_styles
from kys3b.shards import shard_path

N_SHARDS = 12
ROWS = 50


class TestMaxShardsBound(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = Path(self._d.name) / "claims"

    def tearDown(self):
        self._d.cleanup()

    def test_max_shards_1_takes_exactly_one(self):
        cd = ClaimDir(self.dir)
        got = list(cd.iter_available(range(N_SHARDS), lambda s: False, max_shards=1))
        self.assertEqual(len(got), 1)

    def test_max_shards_n_takes_at_most_n(self):
        for n in (1, 3, 7):
            with tempfile.TemporaryDirectory() as d:
                cd = ClaimDir(Path(d) / "c")
                got = list(cd.iter_available(range(N_SHARDS), lambda s: False, max_shards=n))
                self.assertEqual(len(got), n)

    def test_absent_bound_means_production_behaviour(self):
        cd = ClaimDir(self.dir)
        got = list(cd.iter_available(range(N_SHARDS), lambda s: False))
        self.assertEqual(len(got), N_SHARDS, "production must still drain every shard")

    def test_bound_larger_than_the_work_is_harmless(self):
        cd = ClaimDir(self.dir)
        got = list(cd.iter_available(range(3), lambda s: False, max_shards=99))
        self.assertEqual(len(got), 3)

    def test_a_bounded_worker_leaves_the_rest_claimable(self):
        """The shards a smoke job did not take must be untouched for production."""
        a = ClaimDir(self.dir)
        taken = []
        for s in a.iter_available(range(N_SHARDS), lambda s: False, max_shards=2):
            taken.append(s)
            a.release(s)
        b = ClaimDir(self.dir)
        rest = list(b.iter_available(range(N_SHARDS), lambda s: False))
        self.assertEqual(len(rest), N_SHARDS, "released shards must still be claimable")
        self.assertEqual(len(taken), 2)

    def test_claims_are_released_so_a_bounded_worker_blocks_nothing(self):
        a = ClaimDir(self.dir)
        for s in a.iter_available(range(N_SHARDS), lambda s: False, max_shards=1):
            a.release(s)
        self.assertEqual(list(self.dir.glob("*.claim")), [], "no claim left behind")


class TestSmokeOutputIsolation(unittest.TestCase):
    """A partial smoke shard must never be mistaken for finished production work."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        root = Path(self._d.name)
        self.prod = root / "p1"
        self.smoke = root / "_smoke" / "p1"
        self.prod.mkdir(parents=True)
        self.smoke.mkdir(parents=True)
        self.index = {"shards": [{"rows": ROWS} for _ in range(N_SHARDS)]}

    def tearDown(self):
        self._d.cleanup()

    def _write(self, d, k, n_rows):
        atomic_write_table(
            pa.table({
                "doc_id": pa.array(np.arange(n_rows, dtype=np.int64), type=pa.int64()),
                "rewritten": pa.array(["x"] * n_rows, type=pa.large_string()),
                "rewritten_tokens": pa.array(np.ones(n_rows, np.int32), type=pa.int32()),
                "status": pa.array(np.full(n_rows, 2, np.int8), type=pa.int8()),
            }),
            shard_path(d, k),
        )

    def _done(self, s):
        """The worker's completion rule: judged on the PRODUCTION output only."""
        p = shard_path(self.prod, s)
        if not p.exists():
            return False
        try:
            return parquet_rows(p) == self.index["shards"][s]["rows"]
        except Exception:
            return False

    def test_partial_smoke_output_does_not_mark_a_shard_done(self):
        self._write(self.smoke, 0, 7)  # 7 of 50 rows, in the smoke directory
        self.assertFalse(self._done(0), "a smoke shard must leave production work outstanding")
        self.assertFalse(shard_path(self.prod, 0).exists())

    def test_a_partial_production_shard_is_not_counted_done(self):
        self._write(self.prod, 1, 7)
        self.assertFalse(self._done(1), "row count must match the source shard exactly")

    def test_a_full_production_shard_from_a_bounded_run_is_reusable(self):
        """--max-shards without --smoke-rows produces ordinary finished work."""
        self._write(self.prod, 2, ROWS)
        self.assertTrue(self._done(2))
        with tempfile.TemporaryDirectory() as d:
            cd = ClaimDir(Path(d) / "c")
            got = list(cd.iter_available(range(N_SHARDS), self._done))
            self.assertNotIn(2, got, "a completed shard must be skipped by production")
            self.assertEqual(len(got), N_SHARDS - 1)


class TestSmokeStyleFidelity(unittest.TestCase):
    def test_smoke_rows_keep_their_production_wrap_styles(self):
        """Truncating to N rows must not change which style row i gets.

        The worker draws the FULL shard and slices, which is correct by construction regardless
        of how the generator buffers.  (Empirically `integers(0, 4, n)` also happens to be
        prefix-stable on this NumPy, so a short draw gives the same prefix -- but the worker does
        not rely on that, because it is an implementation detail of the bounded-integer path
        rather than a documented guarantee.)
        """
        full = assign_wrap_styles(5, 10_000)
        for n in (1, 7, 64, 500):
            self.assertEqual(
                assign_wrap_styles(5, 10_000)[:n], full[:n],
                "slicing the full draw must reproduce production styles",
            )
        # what the worker actually does, spelled out
        n = 64
        sliced = assign_wrap_styles(5, 10_000)[:n]
        self.assertEqual(len(sliced), n)
        self.assertEqual(sliced, full[:n])

    def test_prefix_stability_is_observed_but_not_depended_on(self):
        """Documents the current NumPy behaviour so a future change is visible, not silent."""
        self.assertEqual(
            assign_wrap_styles(5, 64), assign_wrap_styles(5, 10_000)[:64],
            "NumPy's bounded-integer draw is prefix-stable here; if this ever fails, the "
            "worker is still correct because it slices the full draw",
        )


if __name__ == "__main__":
    unittest.main()
