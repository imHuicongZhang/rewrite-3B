"""Claim-directory work distribution: exclusivity, stale recovery, cooperation, idempotence."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from kys3b.claims import ClaimDir


class TestClaims(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = Path(self._d.name) / "claims"

    def tearDown(self):
        self._d.cleanup()

    def test_claim_is_exclusive(self):
        a = ClaimDir(self.dir)
        b = ClaimDir(self.dir)
        self.assertTrue(a.try_claim(3))
        self.assertFalse(b.try_claim(3), "a live claim must not be stealable")

    def test_release_frees_the_shard(self):
        a = ClaimDir(self.dir)
        b = ClaimDir(self.dir)
        self.assertTrue(a.try_claim(1))
        a.release(1)
        self.assertTrue(b.try_claim(1))

    def test_stale_claim_is_reclaimed(self):
        a = ClaimDir(self.dir, stale_seconds=0)
        self.assertTrue(a.try_claim(5))
        time.sleep(0.01)
        b = ClaimDir(self.dir, stale_seconds=0)
        self.assertTrue(b.try_claim(5), "a claim past its heartbeat window must be reclaimable")

    def test_unreadable_claim_is_judged_by_mtime_not_assumed_stale(self):
        """An unparseable claim is NOT automatically stealable.

        Assuming "cannot parse => stale" would let a second worker steal a shard that was just
        legitimately claimed, because on a filesystem without hard links the claim path exists
        for a moment before its payload lands.  A corrupt claim is judged by the same age rule
        as any other, from its mtime.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / "shard_00007.claim"
        f.write_text("{not json")
        self.assertFalse(
            ClaimDir(self.dir, stale_seconds=3600).try_claim(7),
            "a freshly written (if corrupt) claim must not be stealable",
        )
        os.utime(f, (0, 0))  # now genuinely old
        self.assertTrue(
            ClaimDir(self.dir, stale_seconds=3600).try_claim(7),
            "an old corrupt claim must be reclaimable",
        )

    def test_heartbeat_keeps_a_claim_alive(self):
        a = ClaimDir(self.dir, stale_seconds=3600)
        a.try_claim(2)
        before = json.loads((self.dir / "shard_00002.claim").read_text())["heartbeat"]
        time.sleep(0.02)
        a.heartbeat(2)
        after = json.loads((self.dir / "shard_00002.claim").read_text())["heartbeat"]
        self.assertGreater(after, before)
        self.assertFalse(ClaimDir(self.dir, stale_seconds=3600).try_claim(2))

    def test_completed_shards_are_never_reclaimed(self):
        a = ClaimDir(self.dir)
        done = {0, 1, 2}
        got = list(a.iter_available(range(5), lambda s: s in done))
        self.assertEqual(got, [3, 4], "an existing output means the shard is finished")

    def test_two_workers_partition_the_work_without_overlap(self):
        a = ClaimDir(self.dir)
        b = ClaimDir(self.dir)
        done: set = set()

        def is_done(s):
            return s in done

        ga, gb = [], []
        ia = a.iter_available(range(10), is_done)
        ib = b.iter_available(range(10), is_done)
        # interleave the two generators, completing each shard as it is taken
        for _ in range(10):
            for gen, out, cl in ((ia, ga, a), (ib, gb, b)):
                try:
                    s = next(gen)
                except StopIteration:
                    continue
                out.append(s)
                done.add(s)
                cl.release(s)
        self.assertEqual(set(ga) | set(gb), set(range(10)), "all shards must be covered")
        self.assertEqual(set(ga) & set(gb), set(), "no shard may be processed twice")

    def test_status_counts(self):
        a = ClaimDir(self.dir, stale_seconds=3600)
        a.try_claim(4)
        st = a.status(range(6), lambda s: s in {0, 1})
        self.assertEqual(st["done"], 2)
        self.assertEqual(st["in_flight"], 1)
        self.assertEqual(st["pending"], 3)
        self.assertEqual(st["stale"], 0)

    def test_identity_records_the_slurm_context(self):
        os.environ["SLURM_ARRAY_TASK_ID"] = "17"
        os.environ["SLURM_JOB_ID"] = "999999"
        try:
            i = ClaimDir(self.dir).identity()
            self.assertEqual(i["worker"], "17")
            self.assertEqual(i["jobid"], "999999")
            self.assertEqual(i["pid"], os.getpid())
        finally:
            del os.environ["SLURM_ARRAY_TASK_ID"], os.environ["SLURM_JOB_ID"]


if __name__ == "__main__":
    unittest.main()
