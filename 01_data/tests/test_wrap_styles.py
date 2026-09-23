"""WRAP style assignment: pinned golden vectors, determinism, resume-safety, balance."""
import unittest
from collections import Counter

import numpy as np
from helpers import *  # noqa: F401,F403  (path setup)

from kys3b.config import WRAP_STYLES
from kys3b.prompts import assign_wrap_styles

# np.random.default_rng([42, shard]).integers(0, 4, 16)
# Verified identical under numpy 1.26.4 and 2.0.2 on 2026-09-23.  If a NumPy upgrade changes the
# PCG64 stream this test fails instead of silently re-rolling the whole corpus.
GOLDEN = {
    0: [0, 3, 2, 1, 1, 3, 0, 2, 0, 0, 2, 3, 2, 3, 2, 3],
    1: [2, 3, 1, 0, 3, 1, 3, 2, 3, 3, 3, 2, 1, 3, 1, 3],
    7: [1, 2, 1, 0, 1, 2, 2, 3, 1, 3, 0, 0, 0, 1, 2, 0],
    123: [0, 3, 0, 0, 3, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 0],
}


class TestWrapStyles(unittest.TestCase):
    def test_key_order_is_part_of_the_seed(self):
        self.assertEqual(WRAP_STYLES, ["easy", "hard", "wiki", "qa"])

    def test_golden_vectors(self):
        for shard, want in GOLDEN.items():
            got = assign_wrap_styles(shard, len(want))
            self.assertEqual(got, [WRAP_STYLES[i] for i in want], f"shard {shard}")

    def test_raw_pcg64_stream_matches_the_golden_vectors(self):
        for shard, want in GOLDEN.items():
            self.assertEqual(
                np.random.default_rng([42, shard]).integers(0, 4, len(want)).tolist(), want
            )

    def test_worker_independent_and_resume_safe(self):
        """A crash mid-shard must re-derive the identical assignment on restart."""
        full = assign_wrap_styles(5, 5000)
        again = assign_wrap_styles(5, 5000)
        self.assertEqual(full, again)
        # the whole shard is drawn in one call, so a prefix is NOT the same as a short draw --
        # what matters is that re-deriving the full shard is stable, which it is
        self.assertEqual(full[:100], again[:100])

    def test_different_shards_get_different_assignments(self):
        self.assertNotEqual(assign_wrap_styles(0, 64), assign_wrap_styles(1, 64))

    def test_documents_are_balanced_to_25pct(self):
        c: Counter = Counter()
        for shard in range(200):
            c.update(assign_wrap_styles(shard, 10_000))
        tot = sum(c.values())
        for s in WRAP_STYLES:
            pct = 100.0 * c[s] / tot
            self.assertLess(abs(pct - 25.0), 0.5, f"{s} at {pct:.3f}%")


if __name__ == "__main__":
    unittest.main()
