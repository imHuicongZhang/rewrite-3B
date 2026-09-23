"""A long shard must not become reclaimable while its owner is still generating."""
import json
import os
import threading
import time
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403  (sys.path setup)

from kys3b.claims import ClaimDir, Heartbeat
from kys3b.config import Config


class TestHeartbeatKeepsAClaimLive(unittest.TestCase):
    """Short windows stand in for production's 300 s heartbeat / 1800 s staleness."""

    STALE = 0.30      # claim dies after 0.30 s without a refresh
    INTERVAL = 0.05   # refreshed every 0.05 s

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = Path(self._d.name) / "claims"

    def tearDown(self):
        self._d.cleanup()

    def _hb(self, p):
        return json.loads(p.read_text())["heartbeat"]

    def test_claim_survives_well_beyond_stale_seconds_while_working(self):
        owner = ClaimDir(self.dir, stale_seconds=self.STALE)
        self.assertTrue(owner.try_claim(1))
        path = owner._path(1)
        with Heartbeat(owner, 1, self.INTERVAL) as hb:
            deadline = time.time() + self.STALE * 6      # 6x the staleness window
            while time.time() < deadline:
                self.assertFalse(
                    owner._is_stale(path),
                    "the claim went stale while its owner was still working",
                )
                time.sleep(self.INTERVAL / 2)
        self.assertGreater(hb.beats, 3, "the heartbeat thread never ran")
        self.assertEqual(hb.errors, 0)

    def test_another_worker_cannot_reclaim_while_the_owner_is_working(self):
        owner = ClaimDir(self.dir, stale_seconds=self.STALE)
        self.assertTrue(owner.try_claim(2))
        with Heartbeat(owner, 2, self.INTERVAL):
            deadline = time.time() + self.STALE * 5
            attempts = 0
            while time.time() < deadline:
                thief = ClaimDir(self.dir, stale_seconds=self.STALE)
                self.assertFalse(thief.try_claim(2), "a working owner's shard was stolen")
                attempts += 1
                time.sleep(self.INTERVAL / 2)
        self.assertGreater(attempts, 5)
        self.assertEqual(json.loads(owner._path(2).read_text())["pid"], os.getpid())

    def test_heartbeat_stops_when_the_shard_finishes(self):
        owner = ClaimDir(self.dir, stale_seconds=self.STALE)
        owner.try_claim(3)
        with Heartbeat(owner, 3, self.INTERVAL) as hb:
            time.sleep(self.INTERVAL * 4)
        self.assertFalse(hb.alive, "the heartbeat thread outlived the context")
        last = self._hb(owner._path(3))
        time.sleep(self.INTERVAL * 4)
        self.assertEqual(self._hb(owner._path(3)), last, "the claim is still being refreshed")

    def test_stale_recovery_still_works_after_the_owner_really_disappears(self):
        owner = ClaimDir(self.dir, stale_seconds=self.STALE)
        owner.try_claim(4)
        with Heartbeat(owner, 4, self.INTERVAL):
            time.sleep(self.INTERVAL * 3)
        # the owner is gone; nobody refreshes the claim any more
        time.sleep(self.STALE * 1.5)
        self.assertTrue(
            ClaimDir(self.dir, stale_seconds=self.STALE).try_claim(4),
            "a genuinely abandoned claim must still be reclaimable",
        )

    def test_thread_is_joined_even_when_the_body_raises(self):
        owner = ClaimDir(self.dir, stale_seconds=self.STALE)
        owner.try_claim(5)
        hb = Heartbeat(owner, 5, self.INTERVAL)
        before = threading.active_count()
        with self.assertRaises(ValueError):
            with hb:
                time.sleep(self.INTERVAL * 2)
                raise ValueError("boom")
        self.assertFalse(hb.alive, "the thread leaked after an exception")
        for _ in range(50):
            if threading.active_count() <= before:
                break
            time.sleep(0.01)
        self.assertLessEqual(threading.active_count(), before, "thread count grew")

    def test_never_refreshes_a_claim_this_process_no_longer_owns(self):
        owner = ClaimDir(self.dir, stale_seconds=self.STALE)
        owner.try_claim(6)
        path = owner._path(6)
        with Heartbeat(owner, 6, self.INTERVAL):
            time.sleep(self.INTERVAL * 2)
            # simulate a legitimate reclaim by someone else
            foreign = json.dumps(
                dict(worker="x", array_job="999", host="other", jobid="999", pid=424242,
                     heartbeat=time.time())
            )
            path.write_text(foreign)
            time.sleep(self.INTERVAL * 4)
            self.assertEqual(
                path.read_text(), foreign,
                "the heartbeat must not touch another worker's claim",
            )

    def test_interval_zero_disables_the_thread(self):
        owner = ClaimDir(self.dir, stale_seconds=self.STALE)
        owner.try_claim(7)
        with Heartbeat(owner, 7, 0) as hb:
            time.sleep(0.05)
        self.assertFalse(hb.alive)
        self.assertEqual(hb.beats, 0)

    def test_heartbeat_errors_are_counted_not_swallowed(self):
        owner = ClaimDir(self.dir, stale_seconds=self.STALE)
        owner.try_claim(8)
        boom = RuntimeError("fs down")

        def bad(_shard):
            raise boom

        owner.heartbeat = bad
        with Heartbeat(owner, 8, self.INTERVAL) as hb:
            time.sleep(self.INTERVAL * 4)
        self.assertGreater(hb.errors, 0)
        self.assertIs(hb.last_error, boom)


class TestHeartbeatObservability(unittest.TestCase):
    """A beat must be counted only when the claim was ACTUALLY refreshed."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = Path(self._d.name) / "claims"

    def tearDown(self):
        self._d.cleanup()

    def test_successful_heartbeat_returns_true_and_moves_the_timestamp(self):
        cd = ClaimDir(self.dir, stale_seconds=3600)
        self.assertTrue(cd.try_claim(1))
        p = cd._path(1)
        before = json.loads(p.read_text())["heartbeat"]
        time.sleep(0.01)
        self.assertTrue(cd.heartbeat(1), "a real refresh must report success")
        self.assertGreater(json.loads(p.read_text())["heartbeat"], before)

    def test_heartbeat_on_a_foreign_claim_returns_false_and_changes_nothing(self):
        cd = ClaimDir(self.dir, stale_seconds=3600)
        self.dir.mkdir(parents=True, exist_ok=True)
        foreign = json.dumps(
            dict(worker="x", array_job="9", host="other", jobid="9", pid=424242,
                 heartbeat=time.time())
        )
        (self.dir / "shard_00002.claim").write_text(foreign)
        self.assertFalse(cd.heartbeat(2), "refreshing a foreign claim must report failure")
        self.assertEqual((self.dir / "shard_00002.claim").read_text(), foreign)

    def test_heartbeat_on_a_missing_claim_returns_false(self):
        cd = ClaimDir(self.dir, stale_seconds=3600)
        self.assertFalse(cd.heartbeat(3))
        self.assertFalse(cd._path(3).exists(), "must not conjure a claim out of nothing")

    def test_heartbeat_on_a_corrupt_claim_returns_false_and_does_not_adopt_it(self):
        cd = ClaimDir(self.dir, stale_seconds=3600)
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / "shard_00004.claim"
        f.write_text("{not json")
        self.assertFalse(cd.heartbeat(4))
        self.assertEqual(f.read_text(), "{not json", "an unverifiable claim must not be adopted")

    def test_filesystem_failure_raises_instead_of_looking_successful(self):
        """The whole point: an unwritten claim must never be mistaken for a fresh one."""
        cd = ClaimDir(self.dir, stale_seconds=3600)
        cd.try_claim(5)
        real_replace = os.replace
        calls = {"n": 0}

        def boom(a, b):
            calls["n"] += 1
            raise OSError("simulated filesystem failure")

        os.replace = boom
        try:
            with self.assertRaises(OSError):
                cd.heartbeat(5)
        finally:
            os.replace = real_replace
        self.assertEqual(calls["n"], 1)
        self.assertEqual(list(self.dir.glob("*.claim.hb.*")), [], "scratch file left behind")

    def test_monitor_separates_real_beats_from_lost_ownership(self):
        cd = ClaimDir(self.dir, stale_seconds=3600)
        cd.try_claim(6)
        with Heartbeat(cd, 6, 0.02) as hb:
            time.sleep(0.07)
            beats_while_owned = hb.beats
            self.assertGreater(beats_while_owned, 0)
            self.assertEqual(hb.lost, 0)
            self.assertTrue(hb.healthy)
            # a legitimate reclaim by someone else
            cd._path(6).write_text(
                json.dumps(dict(worker="x", array_job="9", host="other", jobid="9",
                                pid=424242, heartbeat=time.time()))
            )
            time.sleep(0.09)
            self.assertGreater(hb.lost, 0, "lost ownership must be counted, not counted as a beat")
            self.assertEqual(hb.beats, beats_while_owned, "no beats after ownership was lost")
            self.assertFalse(hb.healthy)
        self.assertEqual(hb.errors, 0, "losing ownership is not an error")

    def test_monitor_counts_filesystem_failures_as_errors_not_beats(self):
        cd = ClaimDir(self.dir, stale_seconds=3600)
        cd.try_claim(7)
        boom = OSError("fs down")

        def bad(_shard):
            raise boom

        cd.heartbeat = bad
        with Heartbeat(cd, 7, 0.02) as hb:
            time.sleep(0.09)
        self.assertGreater(hb.errors, 0)
        self.assertEqual(hb.beats, 0, "a failed refresh must never be counted as a beat")
        self.assertIs(hb.last_error, boom)
        self.assertFalse(hb.healthy)


class TestOwnershipBeforeCommit(unittest.TestCase):
    """`owns()` is the gate the worker checks immediately before writing the output shard."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = Path(self._d.name) / "claims"

    def tearDown(self):
        self._d.cleanup()

    def test_owner_may_commit(self):
        cd = ClaimDir(self.dir, stale_seconds=3600)
        self.assertTrue(cd.try_claim(1))
        self.assertTrue(cd.owns(1))

    def test_a_worker_whose_claim_was_replaced_may_not_commit(self):
        cd = ClaimDir(self.dir, stale_seconds=3600)
        cd.try_claim(2)
        self.assertTrue(cd.owns(2))
        cd._path(2).write_text(
            json.dumps(dict(worker="x", array_job="9", host="other", jobid="9",
                            pid=424242, heartbeat=time.time()))
        )
        self.assertFalse(cd.owns(2), "the old owner must be refused the commit")

    def test_a_missing_or_corrupt_claim_is_not_ownership(self):
        cd = ClaimDir(self.dir, stale_seconds=3600)
        self.assertFalse(cd.owns(3), "no claim at all is not ownership")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "shard_00004.claim").write_text("{not json")
        self.assertFalse(cd.owns(4), "an unverifiable claim is not ownership")

    def test_the_old_owner_cannot_delete_or_overwrite_the_new_owners_claim(self):
        cd = ClaimDir(self.dir, stale_seconds=3600)
        cd.try_claim(5)
        new_owner = json.dumps(
            dict(worker="x", array_job="9", host="other", jobid="9", pid=424242,
                 heartbeat=time.time())
        )
        cd._path(5).write_text(new_owner)
        cd.release(5)                                   # the old owner tidies up on its way out
        self.assertTrue(cd._path(5).exists(), "release() deleted the new owner's claim")
        self.assertEqual(cd._path(5).read_text(), new_owner)
        self.assertFalse(cd.heartbeat(5), "heartbeat touched the new owner's claim")
        self.assertEqual(cd._path(5).read_text(), new_owner)

    def test_the_worker_gate_is_wired_and_uses_owns(self):
        """The pre-commit check exists in the worker and calls claims.owns(shard)."""
        src = (Path(__file__).resolve().parents[1] / "src" / "kys3b" / "rewrite"
               / "worker.py").read_text()
        i_check = src.index("claims.owns(shard)")
        i_write = src.index("atomic_write_table(pa.table(cols), shard_path(out_dir, shard))")
        self.assertLess(i_check, i_write, "the ownership check must precede the output write")
        self.assertIn("check(\n        claims.owns(shard),", src)


class TestProductionHeartbeatConfig(unittest.TestCase):
    def test_interval_is_well_inside_the_staleness_window(self):
        c = Config.load()
        self.assertEqual(c.heartbeat_seconds, 300)
        self.assertEqual(c.claim_stale_seconds, 1800)
        self.assertLessEqual(
            c.heartbeat_seconds * 3, c.claim_stale_seconds,
            "at least three refreshes may be missed before a claim is judged dead",
        )

    def test_config_rejects_an_interval_too_close_to_the_window(self):
        from kys3b.io import Stop

        c = Config.load()
        c.budgets["sharding"]["heartbeat_seconds"] = 900  # 900*3 > 1800
        with self.assertRaises(Stop):
            c.check_invariants()


if __name__ == "__main__":
    unittest.main()
