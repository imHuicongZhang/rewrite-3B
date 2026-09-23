"""Shard geometry (the WRAP-style fingerprint) and llama-2 token counting."""
import unittest

import numpy as np
from helpers import *  # noqa: F401,F403  (sys.path setup)

from kys3b.config import Config
from kys3b.pool import doc_id_to_shard, shard_offset, shard_rows
from kys3b.shards import plan_shards
from kys3b.tokens import BOS, train_len


class TestShardGeometry(unittest.TestCase):
    def test_plan_shards_is_a_pure_function_of_the_sorted_id_set(self):
        ids = np.array([9, 1, 5, 3, 7, 2], dtype=np.int64)
        a = plan_shards(ids, 2)
        b = plan_shards(ids[::-1], 2)
        self.assertEqual([list(x) for x in a], [list(x) for x in b])
        self.assertEqual([list(x) for x in a], [[1, 2], [3, 5], [7, 9]])

    def test_plan_shards_sizes(self):
        ids = np.arange(25000, dtype=np.int64)
        g = plan_shards(ids, 10000)
        self.assertEqual([len(x) for x in g], [10000, 10000, 5000])

    def test_doc_id_to_shard_round_trip(self):
        cfg = Config.load()
        rf = int(cfg.pool["rows_full"])
        d = np.array([0, 1, rf - 1, rf, rf + 5, 199 * rf + 7], dtype=np.int64)
        sh, row = doc_id_to_shard(cfg, d)
        self.assertTrue(np.array_equal(sh, np.array([0, 0, 0, 1, 1, 199])))
        self.assertTrue(np.array_equal(row, np.array([0, 1, rf - 1, 0, 5, 7])))
        self.assertTrue(np.array_equal(sh.astype(np.int64) * rf + row, d))

    def test_last_shard_is_short(self):
        cfg = Config.load()
        self.assertEqual(shard_rows(cfg, 0), 500_000)
        self.assertEqual(shard_rows(cfg, 198), 500_000)
        self.assertEqual(shard_rows(cfg, 199), 449_162)
        self.assertEqual(shard_offset(cfg, 199), 199 * 500_000)

    def test_total_matches_the_published_row_count(self):
        cfg = Config.load()
        total = sum(shard_rows(cfg, i) for i in range(int(cfg.pool["n_shards"])))
        self.assertEqual(total, 99_949_162)
        self.assertEqual(total, cfg.pool["n_docs"])


class TestTokenConvention(unittest.TestCase):
    def test_train_length_adds_one_bos(self):
        self.assertEqual(BOS, 1)
        self.assertEqual(train_len(0), 1)
        self.assertEqual(train_len(1234), 1235)

    def test_count_batch_recipe(self):
        cfg = Config.load()
        try:
            from kys3b.tokens import count_batch, load_llama2

            tok = load_llama2(str(cfg.model("llama2")), 32000)
        except Exception as e:
            raise unittest.SkipTest(f"llama-2 tokenizer unavailable: {e}")
        texts = ["", None, "Hello world", "The quick brown fox jumps over the lazy dog."]
        out = count_batch(tok, texts)
        self.assertEqual(out[0], 0, "empty text -> 0")
        self.assertEqual(out[1], 0, "None -> 0")
        self.assertGreater(out[2], 0)
        self.assertGreater(out[3], out[2])
        # no specials: re-encoding with add_special_tokens=False must agree exactly
        for t, n in zip(texts, out):
            if t:
                self.assertEqual(len(tok(t, add_special_tokens=False).input_ids), n)


if __name__ == "__main__":
    unittest.main()
