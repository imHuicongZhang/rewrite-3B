"""Atomic writes and the bucketed document shuffle (the shuffle-parity contract)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from helpers import *  # noqa: F401,F403

from kys3b.io import (
    Stop,
    atomic_write_table,
    bucketed_shuffle,
    paired_wiki_status,
    parquet_rows,
)


def _make(dirpath, n_files=6, rows=500, seed=0):
    rng = np.random.default_rng(seed)
    total = 0
    for i in range(n_files):
        ids = np.arange(total, total + rows, dtype=np.int64)
        total += rows
        t = pa.table(
            {
                "orig_doc_id": pa.array(ids, type=pa.int64()),
                "text": pa.array([f"doc {int(x)}" for x in ids], type=pa.large_string()),
                "source_prompt": pa.array(["original"] * rows, type=pa.large_string()),
                "tokens_llama2": pa.array(rng.integers(5, 50, rows), type=pa.int32()),
            }
        )
        atomic_write_table(t, dirpath / f"in_{i:05d}.parquet")
    return total


class TestAtomicWrite(unittest.TestCase):
    def test_no_tmp_left_behind_on_success(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            atomic_write_table(pa.table({"a": pa.array([1, 2, 3])}), d / "x.parquet")
            self.assertTrue((d / "x.parquet").exists())
            self.assertEqual(list(d.glob("*.tmp")), [])
            self.assertEqual(parquet_rows(d / "x.parquet"), 3)

    def test_tmp_cleaned_up_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            bad = pa.table({"a": pa.array([1])})
            try:
                atomic_write_table(bad, d / "sub" / "y.parquet", compression="not-a-codec")
            except Exception:
                pass
            self.assertEqual(list((d / "sub").glob("*.tmp")), [], "a stale .tmp was left behind")


class TestBucketedShuffle(unittest.TestCase):
    def test_rows_conserved_and_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "in"
            src.mkdir()
            total = _make(src, n_files=6, rows=500)
            paths = sorted(src.glob("*.parquet"))

            r1 = bucketed_shuffle(paths, d / "o1", d / "t1", seed=42, rows_per_out_shard=700,
                                  n_buckets=8)
            r2 = bucketed_shuffle(paths, d / "o2", d / "t2", seed=42, rows_per_out_shard=700,
                                  n_buckets=8)
            self.assertEqual(r1["rows"], total)
            self.assertEqual(r2["rows"], total)

            def read(dirp):
                ids = []
                for p in sorted(dirp.glob("part_*.parquet")):
                    ids.append(
                        pq.read_table(p, columns=["orig_doc_id"])
                        .column("orig_doc_id").to_numpy(zero_copy_only=False)
                    )
                return np.concatenate(ids)

            a, b = read(d / "o1"), read(d / "o2")
            self.assertTrue(np.array_equal(a, b), "the shuffle must be deterministic at seed 42")
            self.assertTrue(np.array_equal(np.sort(a), np.arange(total)), "no row lost or dupl.")
            self.assertFalse(np.array_equal(a, np.arange(total)), "rows must actually be shuffled")

    def test_different_seed_gives_a_different_order(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "in"
            src.mkdir()
            _make(src, n_files=4, rows=300)
            paths = sorted(src.glob("*.parquet"))
            bucketed_shuffle(paths, d / "a", d / "ta", seed=42, rows_per_out_shard=500, n_buckets=4)
            bucketed_shuffle(paths, d / "b", d / "tb", seed=7, rows_per_out_shard=500, n_buckets=4)

            def read(dirp):
                return np.concatenate([
                    pq.read_table(p, columns=["orig_doc_id"])
                    .column("orig_doc_id").to_numpy(zero_copy_only=False)
                    for p in sorted(dirp.glob("part_*.parquet"))
                ])

            self.assertFalse(np.array_equal(read(d / "a"), read(d / "b")))

    def test_bucket_temporaries_are_removed(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "in"
            src.mkdir()
            _make(src, n_files=3, rows=200)
            bucketed_shuffle(sorted(src.glob("*.parquet")), d / "o", d / "tmp",
                             seed=42, rows_per_out_shard=250, n_buckets=5)
            self.assertEqual(list((d / "tmp").glob("bucket_*.parquet")), [])

    def test_output_shards_are_uniform_except_the_last(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "in"
            src.mkdir()
            total = _make(src, n_files=5, rows=400)
            bucketed_shuffle(sorted(src.glob("*.parquet")), d / "o", d / "t",
                             seed=42, rows_per_out_shard=500, n_buckets=6)
            rows = [parquet_rows(p) for p in sorted((d / "o").glob("part_*.parquet"))]
            self.assertEqual(sum(rows), total)
            self.assertTrue(all(r == 500 for r in rows[:-1]), rows)
            self.assertLessEqual(rows[-1], 500)


class TestPairedWikiStatus(unittest.TestCase):
    def test_fast_path_identical_order(self):
        ids = np.arange(10, dtype=np.int64)
        st = np.arange(10, dtype=np.int8) % 3
        out = paired_wiki_status(dict(doc_id=ids, status=st), dict(doc_id=ids, status=st))
        self.assertTrue(np.array_equal(out, st))

    def test_join_fallback_on_permuted_order(self):
        ids = np.arange(10, dtype=np.int64)
        st = (np.arange(10) % 3).astype(np.int8)
        perm = np.array([3, 1, 9, 0, 5, 2, 8, 7, 6, 4])
        out = paired_wiki_status(
            dict(doc_id=ids, status=st), dict(doc_id=ids[perm], status=st[perm])
        )
        self.assertTrue(np.array_equal(out, st[perm]))

    def test_missing_id_is_a_hard_stop(self):
        with self.assertRaises(Stop):
            paired_wiki_status(
                dict(doc_id=np.array([1, 2, 3], np.int64), status=np.zeros(3, np.int8)),
                dict(doc_id=np.array([1, 2, 99], np.int64), status=np.zeros(3, np.int8)),
            )


if __name__ == "__main__":
    unittest.main()
