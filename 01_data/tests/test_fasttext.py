"""The fastText recipe for scoring REWRITTEN text, reproduced from the 1.5B pipeline."""
import unittest

import numpy as np
from helpers import *  # noqa: F401,F403

from kys3b.config import Config
from kys3b.fasttext_score import CLEAN_MAX_CHARS, clean, v2_percentile


class TestClean(unittest.TestCase):
    def test_newlines_and_carriage_returns_become_single_spaces(self):
        self.assertEqual(clean("a\nb\r\nc"), "a b  c")

    def test_no_lowercasing_no_whitespace_collapse_no_stripping(self):
        t = "  Hello   WORLD  <b>tag</b> http://x.y  "
        self.assertEqual(clean(t), t, "only newline/CR replacement and truncation are allowed")

    def test_truncation_at_100k_chars(self):
        self.assertEqual(len(clean("x" * (CLEAN_MAX_CHARS + 500))), CLEAN_MAX_CHARS)

    def test_empty_and_none(self):
        self.assertEqual(clean(""), "")
        self.assertEqual(clean(None), "")


class TestV2Percentile(unittest.TestCase):
    def test_tie_aware_average_rank_over_the_reference(self):
        ref = np.sort(np.array([0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.9], dtype=np.float32))
        n = ref.size
        p = v2_percentile(np.array([0.0, 0.1, 0.2, 0.9], dtype=np.float32), ref, n)
        # 0.0 occupies ranks 1-2 -> mean 1.5 ; 0.1 rank 3 ; 0.2 ranks 4-6 -> mean 5 ; 0.9 rank 7
        self.assertAlmostEqual(float(p[0]), 1.5 / n, places=6)
        self.assertAlmostEqual(float(p[1]), 3.0 / n, places=6)
        self.assertAlmostEqual(float(p[2]), 5.0 / n, places=6)
        self.assertAlmostEqual(float(p[3]), 7.0 / n, places=6)

    def test_monotone_non_decreasing_in_the_raw_score(self):
        """The percentile is a STEP function of the raw score: monotone non-decreasing.

        Two distinct raw scores falling in the same gap of the reference distribution get the
        same percentile, so the map is not strictly increasing and argsort orders can differ at
        those positions.  What is guaranteed -- and what the 1.5B pipeline relied on -- is that
        the percentile never decreases as the raw score rises.  This is why
        `04_filter_top5B_rewrite` sorts by the RAW score and treats the percentile as a
        readability column only.
        """
        rng = np.random.default_rng(0)
        ref = np.sort(rng.random(5000).astype(np.float32))
        raw = np.sort(rng.random(200).astype(np.float32))
        p = v2_percentile(raw, ref, ref.size)
        self.assertTrue(np.all(np.diff(p) >= 0), "percentile decreased as the raw score rose")
        self.assertGreater(np.unique(p).size, 150, "the map should still be nearly injective")

    def test_ranking_by_raw_and_by_percentile_agree_when_there_are_no_step_ties(self):
        ref = np.sort(np.linspace(0.0, 1.0, 100_001, dtype=np.float32))
        raw = np.array([0.9, 0.1, 0.5, 0.77, 0.23], dtype=np.float32)
        p = v2_percentile(raw, ref, ref.size)
        self.assertEqual(
            list(np.argsort(-raw, kind="stable")), list(np.argsort(-p, kind="stable"))
        )


class TestScorerAgainstTheRealModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from kys3b.fasttext_score import FastTextScorer

            cls.s = FastTextScorer(Config.load().model("fasttext"))
        except Exception as e:
            raise unittest.SkipTest(f"fastText model unavailable: {e}")

    def test_scores_are_probabilities(self):
        texts = [
            "The mitochondrion is a double-membrane-bound organelle found in most eukaryotes.",
            "click here!!! buy now cheap pills $$$ visit our site",
            "",
        ]
        s = self.s.score_many(texts)
        self.assertEqual(s.shape, (3,))
        self.assertEqual(float(s[2]), 0.0, "empty text scores exactly 0.0, with no predict call")
        for x in s[:2]:
            self.assertGreaterEqual(float(x), -1e-4)
            self.assertLessEqual(float(x), 1.0 + 1e-4)

    def test_educational_text_outscores_spam(self):
        good = self.s.score_one(
            "Photosynthesis converts light energy into chemical energy stored in glucose. "
            "In plants it occurs in the chloroplasts, where chlorophyll absorbs photons."
        )
        spam = self.s.score_one("CLICK HERE!!! cheap pills buy now $$$$ limited offer")
        self.assertGreater(good, spam)

    def test_newlines_do_not_change_the_score(self):
        a = self.s.score_one("The capital of France is Paris. It lies on the river Seine.")
        b = self.s.score_one("The capital of France is Paris.\nIt lies on the river Seine.")
        self.assertAlmostEqual(a, b, places=4, msg="clean() replaces newlines before predict")

    def test_deterministic(self):
        t = "A deterministic scorer must return the same value every call."
        self.assertEqual(self.s.score_one(t), self.s.score_one(t))


if __name__ == "__main__":
    unittest.main()
