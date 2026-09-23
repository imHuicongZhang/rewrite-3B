import unittest

import numpy as np
from helpers import synthetic_pool

from kys3b.io import Stop
from kys3b.select import fill_to, fill_to_strict, order_desc, rng_children


class TestFillTo(unittest.TestCase):
    def setUp(self):
        self.tok = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        self.order = np.arange(5)

    def test_last_document_kept_whole(self):
        sel, total, over, filled = fill_to(self.order, self.tok, 45)
        self.assertTrue(filled)
        self.assertEqual(list(sel), [0, 1, 2])          # 10+20+30 = 60 >= 45
        self.assertEqual(total, 60)
        self.assertEqual(over, 15)
        self.assertLess(over, self.tok[sel[-1]], "overshoot < the last document's length")

    def test_exact_boundary_stops_immediately(self):
        sel, total, over, filled = fill_to(self.order, self.tok, 30)
        self.assertTrue(filled)
        self.assertEqual(list(sel), [0, 1])
        self.assertEqual(total, 30)
        self.assertEqual(over, 0)

    def test_underfill_reports_not_filled(self):
        sel, total, over, filled = fill_to(self.order, self.tok, 1000)
        self.assertFalse(filled)
        self.assertEqual(total, 150)
        self.assertEqual(over, 150 - 1000)

    def test_empty_order(self):
        sel, total, over, filled = fill_to(np.array([], dtype=np.int64), self.tok, 10)
        self.assertFalse(filled)
        self.assertEqual(total, 0)

    def test_strict_stops_on_underfill(self):
        with self.assertRaises(Stop):
            fill_to_strict(self.order, self.tok, 1000, "test block")

    def test_strict_returns_a_copy_not_a_view(self):
        sel, _, _ = fill_to_strict(self.order, self.tok, 45, "t")
        self.assertIsNone(sel.base, "fill_to_strict must .copy() so the order array can be freed")


class TestOrderDesc(unittest.TestCase):
    def test_descending_with_deterministic_tie_break(self):
        score = np.array([0.5, 0.9, 0.5, 0.1], dtype=np.float32)
        tie = np.array([3, 0, 1, 2], dtype=np.int64)
        idx = np.arange(4)
        out = order_desc(idx, score, tie)
        self.assertEqual(out[0], 1)                 # highest score
        self.assertEqual(list(out[1:3]), [2, 0])    # tie at 0.5 broken by tie ASC
        self.assertEqual(out[3], 3)

    def test_tie_floor_is_broken_deterministically_and_uses_the_whole_tie_group(self):
        pool = synthetic_pool(n=5000)
        tie = rng_children(42)[1].permutation(pool.n).astype(np.int64)
        idx = np.arange(pool.n)
        a = order_desc(idx, pool.ft, tie)
        b = order_desc(idx, pool.ft, tie)
        self.assertTrue(np.array_equal(a, b), "order_desc must be deterministic")
        floor = np.flatnonzero(pool.ft == np.float32(0.0433))
        self.assertGreater(floor.size, 50, "the synthetic pool should have a real tie floor")
        # every tie-floor document must sit in a contiguous block at the end of the order
        pos = np.argsort(a)
        fp = np.sort(pos[floor])
        self.assertEqual(fp[-1], pool.n - 1)
        self.assertTrue(np.array_equal(fp, np.arange(fp[0], pool.n)))

    def test_seed_children_are_stable(self):
        a = [c.integers(0, 1_000_000) for c in rng_children(42)]
        b = [c.integers(0, 1_000_000) for c in rng_children(42)]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
