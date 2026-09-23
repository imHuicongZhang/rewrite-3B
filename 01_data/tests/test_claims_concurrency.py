"""Stale-claim reclamation must have exactly ONE winner, under real process concurrency.

These run real OS processes (spawn, so each has its own pid and its own ClaimDir) hammering the
same stale claim at the same moment.  Correctness must not depend on sleep timing.
"""
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403  (sys.path setup)

from kys3b.claims import ClaimDir

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _reclaim_worker(claim_dir, shard, stale_seconds, barrier, results, idx):
    """Child process: line up on a barrier, then all try to reclaim the same shard at once."""
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    from kys3b.claims import ClaimDir as CD

    cd = CD(Path(claim_dir), stale_seconds=stale_seconds)
    try:
        barrier.wait(timeout=30)
    except Exception:
        pass
    won = cd.try_claim(shard)
    results[idx] = 1 if won else 0
    if won:
        # record who believes they own it, so the parent can cross-check the claim content
        results[idx] = os.getpid()


def _first_claim_worker(claim_dir, shard, barrier, results, idx):
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    from kys3b.claims import ClaimDir as CD

    cd = CD(Path(claim_dir))
    try:
        barrier.wait(timeout=30)
    except Exception:
        pass
    results[idx] = os.getpid() if cd.try_claim(shard) else 0


class TestReclaimSingleWinner(unittest.TestCase):
    N = 16

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = Path(self._d.name) / "claims"
        self.ctx = mp.get_context("spawn")

    def tearDown(self):
        self._d.cleanup()

    def _plant_stale_claim(self, shard: int, age: float = 10_000.0) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"shard_{shard:05d}.claim").write_text(
            json.dumps(
                dict(worker="99", host="deadnode", jobid="", pid=999999,
                     heartbeat=time.time() - age)
            )
        )

    # The window must be LONGER than the wall-clock spread of `spawn` starting N processes.
    # With stale_seconds=1 a winner's own fresh claim ages past the window before the slowest
    # sibling even checks it, so every sibling then reclaims it *legitimately* and the test sees
    # many "winners" while the implementation is behaving correctly.  The planted claim is aged
    # 10,000 s, so a long window still makes it stale.  Correctness must not depend on timing --
    # and neither must the test.
    def _run(self, target, shard, stale_seconds=3600):
        barrier = self.ctx.Barrier(self.N)
        results = self.ctx.Array("l", self.N)
        args_extra = (stale_seconds,) if target is _reclaim_worker else ()
        procs = [
            self.ctx.Process(
                target=target,
                args=(str(self.dir), shard, *args_extra, barrier, results, i),
            )
            for i in range(self.N)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            self.assertIsNotNone(p.exitcode, "a child hung")
            self.assertEqual(p.exitcode, 0, "a child crashed")
        return [results[i] for i in range(self.N)]

    def test_exactly_one_process_reclaims_a_stale_claim(self):
        self._plant_stale_claim(3)
        got = self._run(_reclaim_worker, 3)
        winners = [g for g in got if g != 0]
        self.assertEqual(len(winners), 1, f"expected exactly one winner, got {winners}")
        meta = json.loads((self.dir / "shard_00003.claim").read_text())
        self.assertEqual(meta["pid"], winners[0], "the claim on disk must name the winner")

    def test_exactly_one_process_wins_a_first_claim(self):
        got = self._run(_first_claim_worker, 4)
        winners = [g for g in got if g != 0]
        self.assertEqual(len(winners), 1, f"expected exactly one winner, got {winners}")
        self.assertEqual(
            json.loads((self.dir / "shard_00004.claim").read_text())["pid"], winners[0]
        )

    def test_repeated_rounds_never_double_grant(self):
        """20 independent stale-claim stampedes; every one must have a single winner."""
        for shard in range(20):
            self._plant_stale_claim(shard)
            got = self._run(_reclaim_worker, shard)
            self.assertEqual(
                len([g for g in got if g != 0]), 1, f"shard {shard}: not exactly one winner"
            )

    def test_a_live_claim_is_never_stolen(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "shard_00009.claim").write_text(
            json.dumps(dict(worker="1", host="livenode", jobid="", pid=12345,
                            heartbeat=time.time()))
        )
        got = self._run(_reclaim_worker, 9, stale_seconds=3600)
        self.assertEqual([g for g in got if g != 0], [], "a live claim must not be reclaimable")
        self.assertEqual(
            json.loads((self.dir / "shard_00009.claim").read_text())["pid"], 12345,
            "the live owner's claim must be intact",
        )

    def test_a_dead_reclaim_guard_does_not_block_the_shard_forever(self):
        """A worker killed inside the critical section leaves a guard; the age rule frees it."""
        self._plant_stale_claim(11)
        cd = ClaimDir(self.dir, stale_seconds=3600)
        guard = cd._guard_path(11)
        os.mkdir(guard)  # simulate the crash: guard held, worker gone

        self.assertFalse(
            ClaimDir(self.dir, stale_seconds=3600).try_claim(11),
            "a FRESH guard must block reclaim -- that is the mutual exclusion working",
        )
        os.utime(guard, (0, 0))  # now older than the window
        self.assertTrue(
            ClaimDir(self.dir, stale_seconds=3600).try_claim(11),
            "a guard older than the staleness window must be re-acquirable",
        )
        self.assertFalse(guard.exists(), "the guard must be released after the section")

    def test_live_claim_is_never_moved_away(self):
        """The claim path must never vanish while a live owner holds it.

        An earlier design renamed the claim aside to serialise reclaimers; that momentary absence
        let a first-claim path succeed beside the reclaimer, granting the shard twice.
        """
        a = ClaimDir(self.dir, stale_seconds=3600)
        self.assertTrue(a.try_claim(12))
        path = a._path(12)
        for _ in range(50):
            b = ClaimDir(self.dir, stale_seconds=3600)
            self.assertFalse(b.try_claim(12))
            self.assertTrue(path.exists(), "the live claim disappeared from disk")
        self.assertEqual(json.loads(path.read_text())["pid"], os.getpid())


class TestOwnershipSafety(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = Path(self._d.name) / "claims"

    def tearDown(self):
        self._d.cleanup()

    def test_release_does_not_delete_another_workers_claim(self):
        """A wedged worker whose claim was reclaimed must not delete the new owner's claim."""
        a = ClaimDir(self.dir, stale_seconds=0)
        self.assertTrue(a.try_claim(2))
        # b reclaims it (a's heartbeat is instantly stale)
        b = ClaimDir(self.dir, stale_seconds=0)
        (self.dir / "shard_00002.claim").write_text(
            json.dumps(dict(worker="b", host="other", jobid="", pid=424242,
                            heartbeat=time.time()))
        )
        a.release(2)  # a wakes up late and tries to clean up
        self.assertTrue(
            (self.dir / "shard_00002.claim").exists(),
            "release() must not remove a claim this process does not own",
        )
        self.assertEqual(
            json.loads((self.dir / "shard_00002.claim").read_text())["pid"], 424242
        )
        del b

    def test_heartbeat_does_not_touch_another_workers_claim(self):
        a = ClaimDir(self.dir, stale_seconds=3600)
        self.dir.mkdir(parents=True, exist_ok=True)
        before = json.dumps(
            dict(worker="b", host="other", jobid="", pid=424242, heartbeat=123.0)
        )
        (self.dir / "shard_00005.claim").write_text(before)
        a.heartbeat(5)
        self.assertEqual((self.dir / "shard_00005.claim").read_text(), before)

    def test_heartbeat_tmp_path_is_per_process(self):
        a = ClaimDir(self.dir)
        a.try_claim(6)
        a.heartbeat(6)
        self.assertEqual(list(self.dir.glob("*.claim.hb.*")), [], "no scratch file left behind")

    def test_owner_can_heartbeat_and_keep_the_claim(self):
        a = ClaimDir(self.dir, stale_seconds=3600)
        self.assertTrue(a.try_claim(7))
        t0 = json.loads((self.dir / "shard_00007.claim").read_text())["heartbeat"]
        time.sleep(0.02)
        a.heartbeat(7)
        t1 = json.loads((self.dir / "shard_00007.claim").read_text())["heartbeat"]
        self.assertGreater(t1, t0)
        self.assertFalse(ClaimDir(self.dir, stale_seconds=3600).try_claim(7))


if __name__ == "__main__":
    unittest.main()
