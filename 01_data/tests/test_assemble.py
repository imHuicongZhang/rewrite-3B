"""Assembly: pass-1 first, then the per-arm distill top-up rule; never pad."""
import unittest

import numpy as np
from helpers import pass_arrays

from kys3b.post.assemble import (
    assemble_flat,
    assemble_per_topic,
    fill_desc,
    sort_key_values,
)


class TestFillDesc(unittest.TestCase):
    def test_stable_descending_no_seeded_tie_break(self):
        idx = np.arange(5)
        key = np.array([0.1, 0.9, 0.9, 0.5, 0.2], dtype=np.float32)
        length = np.full(5, 10, dtype=np.int64)
        sel, tot, filled = fill_desc(idx, key, length, 25)
        self.assertTrue(filled)
        self.assertEqual(list(sel), [1, 2, 3], "equal keys keep input order (stable sort)")
        self.assertEqual(tot, 30)

    def test_underfill(self):
        idx = np.arange(3)
        sel, tot, filled = fill_desc(
            idx, np.arange(3, dtype=np.float32), np.full(3, 5, np.int64), 100
        )
        self.assertFalse(filled)
        self.assertEqual(tot, 15)


class TestSortKeys(unittest.TestCase):
    def test_fasttext_key_is_the_fasttext_percentile(self):
        ft = np.array([0.1, 0.9], np.float32)
        out = sort_key_values("fasttext", ft, np.zeros(2, np.float32), np.zeros(2, np.float32))
        self.assertTrue(np.array_equal(out, ft))

    def test_u_key_is_q_plus_half_sqrt_v(self):
        ft = np.array([0.9, 0.2], np.float32)
        fw = np.array([0.1, 0.2], np.float32)
        mb = np.array([0.5, 0.2], np.float32)
        out = sort_key_values("u", ft, fw, mb, 0.5)
        q = ((ft + fw + mb) / 3).astype(np.float32)
        v = (((ft - q) ** 2 + (fw - q) ** 2 + (mb - q) ** 2) / 3).astype(np.float32)
        self.assertTrue(np.allclose(out, q + np.float32(0.5) * np.sqrt(v)))
        self.assertGreater(out[0], q[0], "a disagreeing document gets a bonus")
        self.assertAlmostEqual(float(out[1]), float(q[1]), places=6, msg="no disagreement, no bonus")


class TestAssembleFlat(unittest.TestCase):
    def _pair(self, n=400, seed=1):
        rng = np.random.default_rng(seed)
        ids = np.arange(n)
        st = np.full(n, 2, np.int8)
        st[:5] = 0          # dropped
        st[5:8] = 1         # truncated
        p1 = pass_arrays(ids, st, rng.integers(100, 400, n), key=rng.random(n))
        dis = pass_arrays(ids, st, rng.integers(60, 250, n), key=rng.random(n))
        return p1, dis

    def test_all_pass1_then_key_ordered_distill(self):
        p1, dis = self._pair()
        p1_avail = int(p1["len"][p1["status"] == 2].sum())
        target = p1_avail + 10_000
        r = assemble_flat(p1, dis, target, "pass1_then_key")
        self.assertTrue(r["used_distill"])
        self.assertEqual(r["p1_docs"], int((p1["status"] == 2).sum()), "ALL status==2 pass-1 kept")
        self.assertEqual(r["p1_tokens"], p1_avail)
        self.assertGreaterEqual(r["total_tokens"], target)
        # no status 0/1 row may ever be kept
        self.assertFalse(r["keep_p1"][p1["status"] != 2].any())
        self.assertFalse(r["keep_distill"][dis["status"] != 2].any())
        # the distill rows kept must be the top of the key order
        kept = np.flatnonzero(r["keep_distill"])
        rej = np.flatnonzero((~r["keep_distill"]) & (dis["status"] == 2))
        self.assertGreater(dis["key"][kept].min(), dis["key"][rej].max() - 1e-6)

    def test_pass1_alone_fills_branch(self):
        p1, dis = self._pair()
        target = int(p1["len"][p1["status"] == 2].sum()) // 2
        r = assemble_flat(p1, dis, target, "pass1_then_key")
        self.assertFalse(r["used_distill"], "the 1.5B 'pass1 alone fills B' branch")
        self.assertEqual(r["distill_tokens"], 0)
        self.assertGreaterEqual(r["total_tokens"], target)

    def test_shortfall_is_reported_and_never_padded(self):
        p1, dis = self._pair()
        huge = 10**12
        r = assemble_flat(p1, dis, huge, "pass1_then_key")
        self.assertGreater(r["shortfall"], 0)
        self.assertEqual(
            r["total_tokens"],
            int(p1["len"][p1["status"] == 2].sum() + dis["len"][dis["status"] == 2].sum()),
            "everything available is used, and nothing is invented",
        )

    def test_wrap_distill_topup_uses_no_quality_signal(self):
        """The wrap arm's distill top-up must be a seeded random draw, not key-ordered."""
        p1, dis = self._pair()
        target = int(p1["len"][p1["status"] == 2].sum()) + 8000
        rnd = assemble_flat(p1, dis, target, "pass1_then_random")
        keyed = assemble_flat(p1, dis, target, "pass1_then_key")
        self.assertFalse(
            np.array_equal(rnd["keep_distill"], keyed["keep_distill"]),
            "random and keyed top-ups must differ",
        )
        kept = np.flatnonzero(rnd["keep_distill"])
        rej = np.flatnonzero((~rnd["keep_distill"]) & (dis["status"] == 2))
        # a random draw must NOT be quality-separated
        self.assertLess(
            abs(dis["key"][kept].mean() - dis["key"][rej].mean()), 0.12,
            "the wrap distill draw must carry no quality signal",
        )

    def test_wrap_random_draw_is_seeded_and_reproducible(self):
        p1, dis = self._pair()
        target = int(p1["len"][p1["status"] == 2].sum()) + 8000
        a = assemble_flat(p1, dis, target, "pass1_then_random", seed=42)
        b = assemble_flat(p1, dis, target, "pass1_then_random", seed=42)
        self.assertTrue(np.array_equal(a["keep_distill"], b["keep_distill"]))

    def test_dual_doc_ids_are_kept_as_two_examples(self):
        p1, dis = self._pair()
        target = int(p1["len"][p1["status"] == 2].sum()) + 8000
        r = assemble_flat(p1, dis, target, "pass1_then_key")
        dual = np.intersect1d(p1["doc_id"][r["keep_p1"]], dis["doc_id"][r["keep_distill"]])
        self.assertGreater(dual.size, 0, "distill is NOT deduplicated against pass 1")


class TestAssemblePerTopic(unittest.TestCase):
    def test_policy_a_no_cross_topic_backfill_no_padding(self):
        rng = np.random.default_rng(2)
        n = 2400
        ids = np.arange(n)
        topic = (ids % 24).astype(np.int8)
        st = np.full(n, 2, np.int8)
        # starve topic 0 completely
        st[topic == 0] = 0
        p1 = pass_arrays(ids, st, rng.integers(100, 300, n), key=rng.random(n), topic=topic)
        dis = pass_arrays(ids, st, rng.integers(60, 200, n), key=rng.random(n), topic=topic)
        props = {c: 1 / 24 for c in range(24)}
        target = 200_000
        r = assemble_per_topic(p1, dis, target, props)
        self.assertGreater(r["shortfall"], 0, "a starved topic must produce a reported shortfall")
        self.assertIn(0, r["shortfall_topics"])
        # nothing from topic 0 was kept, and no other topic was used to cover for it
        self.assertEqual(int(r["keep_p1"][topic == 0].sum()), 0)
        for c in range(1, 24):
            got = next(x for x in r["per_topic"] if x["topic"] == c)
            self.assertLessEqual(
                got["tokens"], got["quota_tokens"] + 400,
                f"topic {c} exceeded its quota -- that would be cross-topic backfill",
            )

    def test_within_topic_key_order_is_respected(self):
        rng = np.random.default_rng(6)
        n = 2400
        ids = np.arange(n)
        topic = (ids % 24).astype(np.int8)
        st = np.full(n, 2, np.int8)
        p1 = pass_arrays(ids, st, np.full(n, 50), key=rng.random(n), topic=topic)
        dis = pass_arrays(ids, st, np.full(n, 40), key=rng.random(n), topic=topic)
        props = {c: 1 / 24 for c in range(24)}
        # quota small enough that pass 1 alone over-fills each topic -> key order decides
        r = assemble_per_topic(p1, dis, 24 * 200, props)
        for c in range(24):
            kept = np.flatnonzero(r["keep_p1"] & (topic == c))
            rej = np.flatnonzero((~r["keep_p1"]) & (topic == c))
            if kept.size and rej.size:
                self.assertGreaterEqual(
                    float(p1["key"][kept].min()) + 1e-6, float(p1["key"][rej].max())
                )


if __name__ == "__main__":
    unittest.main()
