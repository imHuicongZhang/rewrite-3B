"""Claim-directory work distribution with race-safe stale recovery (Decision 8).

Replaces the 1.5B `shard_idx % num_workers` assignment.  Modulo assignment strands a dead
worker's shards until that exact array index is resubmitted, which is unacceptable on a
preemptible scavenger queue.  This is BEHAVIOURALLY NEUTRAL: per-document output depends only
on (shard_index, row_index), never on which worker ran the shard.

A claim is a file holding JSON {worker, host, jobid, pid, heartbeat}.  Ownership is granted by
exactly two operations, and both are single-winner filesystem primitives:

  first claim   `os.link(private_tmp, claim)`  -- atomic, fails if the claim exists, and the
                                                  payload is complete BEFORE the path appears
  reclaim       `os.mkdir(claim + ".reclaim")` -- atomic; exactly one process enters the
                                                  reclaim critical section for a shard

The reclaim critical section, in order: acquire the guard; re-read the claim; re-check that it
is still stale (it may have been heart-beaten since the first check); overwrite it in place with
`os.replace`; release the guard.  A loser fails immediately and moves on -- nothing blocks and
nothing sleeps.

Two properties make this safe, and both were established the hard way by
`tests/test_claims_concurrency.py`:

  * **A live claim is never moved.**  An earlier design renamed the stale claim to a private
    path to serialise reclaimers.  That leaves the claim path momentarily ABSENT, so another
    worker's first-claim path can succeed at the same time as the reclaimer -- two owners.  Here
    the claim file stays in place throughout, so the first-claim path can only fire when the
    shard genuinely has no claim.
  * **A claim is never visible half-written.**  `O_CREAT | O_EXCL` followed by a write makes the
    path exist before the payload lands, so a racing worker can read an empty file, fail to
    parse it and judge the claim stale.  Linking a fully-written private file closes that window.

Guard liveness: a worker killed inside the critical section leaves the guard directory behind.
A guard older than `stale_seconds` is removed and re-acquired, using the same age rule as claims
themselves; `mkdir` still admits only one winner, so a concurrent sweep cannot double-grant.

Requires atomic `mkdir`, `link` and `replace`.  WekaFS provides all three (verified).

Completion is recorded by the OUTPUT parquet existing with the expected row count, never by the
claim: an output that exists is never redone, which is what makes a rerun idempotent.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from .io import log

PRIVATE_SUFFIX = ".tmp"   # private, per-process scratch beside a claim


class ClaimDir:
    def __init__(self, directory: Path, stale_seconds: float = 1800):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        # float, not int: `int(0.3)` is 0, which would make every claim instantly stale.  That
        # only bites sub-second windows (tests), but truncating a caller's value silently is the
        # kind of thing that looks fine until it does not.
        self.stale_seconds = float(stale_seconds)
        self._live_jobs: set[str] | None = None
        self._live_jobs_at = 0.0

    # ------------------------------------------------------------------ identity
    @staticmethod
    def identity() -> dict:
        """Who this process is.

        `array_job` is SLURM_ARRAY_JOB_ID (the id shared by every task of one array), falling
        back to SLURM_JOB_ID.  It is what distinguishes the primary jhu2 array from the
        opportunistic scavenger array: both number their tasks from 0, so the task id ALONE is
        not a unique worker identity.
        """
        return dict(
            worker=os.environ.get("SLURM_ARRAY_TASK_ID", "local"),
            array_job=os.environ.get("SLURM_ARRAY_JOB_ID")
            or os.environ.get("SLURM_JOB_ID")
            or "local",
            host=socket.gethostname(),
            jobid=os.environ.get("SLURM_JOB_ID", ""),
            pid=os.getpid(),
        )

    @staticmethod
    def worker_key() -> str:
        """A globally unique worker identifier, safe as a filename component.

        Includes the array job id AND the task id, so primary task 0 and scavenger task 0 never
        collide, plus host and pid so two local runs -- or a requeued task that gets the same
        (array_job, task) with a fresh process -- also stay distinct.  A requeued task therefore
        writes a NEW progress file rather than overwriting the record of the work it did before
        being preempted.
        """
        i = ClaimDir.identity()
        safe = str(i["host"]).replace("/", "_")
        return f"{i['array_job']}_t{i['worker']}_{safe}_p{i['pid']}"

    def _is_mine(self, meta: dict) -> bool:
        me = self.identity()
        return meta.get("pid") == me["pid"] and meta.get("host") == me["host"]

    def _path(self, shard: int) -> Path:
        return self.dir / f"shard_{shard:05d}.claim"

    def _private_path_suffix(self, p: Path, kind: str) -> Path:
        me = self.identity()
        return p.with_name(
            f"{p.name}{PRIVATE_SUFFIX}.{kind}.{me['host']}.{me['pid']}.{time.time_ns()}"
        )

    # ------------------------------------------------------------------ liveness
    def _jobs_alive(self) -> set[str]:
        if self._live_jobs is not None and (time.time() - self._live_jobs_at) < 120:
            return self._live_jobs
        jobs: set[str] = set()
        try:
            out = subprocess.run(
                ["squeue", "-h", "-o", "%A"], capture_output=True, text=True, timeout=30
            )
            if out.returncode == 0:
                jobs = {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}
        except Exception:
            jobs = set()
        self._live_jobs, self._live_jobs_at = jobs, time.time()
        return jobs

    def _meta_is_stale(self, meta: dict | None, path: Path | None = None) -> bool:
        """Staleness from claim CONTENT: heartbeat age, or a Slurm job that is gone.

        An unreadable claim is NOT assumed stale.  On a filesystem without hard links the
        O_EXCL fallback in `_create_exclusive` leaves a momentary empty file, and assuming
        "unparseable => stale" would let a second worker steal a shard that was just claimed.
        Such a claim is judged by its mtime instead -- the same age rule, from the only
        information available.
        """
        if meta is None:
            if path is None:
                return True
            try:
                return (time.time() - path.stat().st_mtime) > self.stale_seconds
            except FileNotFoundError:
                return True
        age = time.time() - float(meta.get("heartbeat", 0))
        if age > self.stale_seconds:
            return True
        jobid = str(meta.get("jobid") or "")
        if jobid:
            alive = self._jobs_alive()
            # only trust squeue when it returned something; an empty set may mean it failed
            if alive and jobid not in alive:
                return True
        return False

    @staticmethod
    def _read(p: Path) -> dict | None:
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _is_stale(self, p: Path) -> bool:
        return self._meta_is_stale(self._read(p), p)

    # ------------------------------------------------------------------ claim / release
    def _create_exclusive(self, p: Path) -> bool:
        """Publish a COMPLETE claim at `p`, or return False because someone else holds it.

        Not `O_CREAT | O_EXCL` + write: that makes the path exist BEFORE the payload lands, so a
        racing worker can read an empty file, fail to parse it, judge the claim stale and reclaim
        a shard that was just legitimately taken.  (That interleaving is real -- the concurrency
        test caught it.)

        Instead the payload is written in full to a private path and then `os.link`ed into place.
        `link` is atomic and fails with FileExistsError if the target exists, so it is a
        single-winner primitive AND the claim is never visible in a partial state.  WekaFS
        supports hard links; if the filesystem does not, we fall back to O_EXCL and rely on the
        mtime-based staleness floor in `_meta_is_stale` to close the same window.
        """
        payload = json.dumps({**self.identity(), "heartbeat": time.time()})
        tmp = self._private_path_suffix(p, "new")
        try:
            tmp.write_text(payload)
            try:
                os.link(tmp, p)
                return True
            except FileExistsError:
                return False
            except OSError:
                # filesystem without hard links: degrade to O_EXCL + immediate write
                try:
                    fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                except FileExistsError:
                    return False
                with os.fdopen(fd, "w") as f:
                    f.write(payload)
                return True
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def _guard_path(self, shard: int) -> Path:
        return self.dir / f"shard_{shard:05d}.reclaim"

    def _acquire_guard(self, guard: Path) -> bool:
        """Enter the reclaim critical section, or fail immediately.  `mkdir` admits one winner."""
        try:
            os.mkdir(guard)
            return True
        except FileExistsError:
            pass
        # A guard older than the staleness window belonged to a worker that died inside the
        # section.  Remove it and retry once; mkdir still admits only one winner, so a race
        # between two sweepers cannot grant twice.
        try:
            age = time.time() - guard.stat().st_mtime
        except FileNotFoundError:
            age = None
        if age is not None and age > self.stale_seconds:
            try:
                os.rmdir(guard)
            except OSError:
                pass
            try:
                os.mkdir(guard)
                return True
            except OSError:
                return False
        return False

    @staticmethod
    def _release_guard(guard: Path) -> None:
        try:
            os.rmdir(guard)
        except OSError:
            pass

    def try_claim(self, shard: int) -> bool:
        """Become the single owner of `shard`, or return False.  Never blocks, never sleeps."""
        p = self._path(shard)

        # ---- fast path: the shard has no claim at all ----
        if self._create_exclusive(p):
            return True

        # ---- a claim exists; only a STALE one may be taken ----
        if not self._is_stale(p):
            return False

        guard = self._guard_path(shard)
        if not self._acquire_guard(guard):
            return False  # another worker is reclaiming this shard right now
        try:
            if not p.exists():
                # released while we were acquiring the guard: fall back to the first-claim
                # primitive, which is still single-winner against concurrent fast-path claimers
                return self._create_exclusive(p)
            if not self._is_stale(p):
                return False  # heart-beaten since our first check; the owner is alive
            # Overwrite in place.  Only guard holders ever modify a claim they do not own, and
            # the path stays present throughout, so no fast-path claimer can slip in beside us.
            tmp = self._private_path_suffix(p, "reclaim")
            try:
                tmp.write_text(json.dumps({**self.identity(), "heartbeat": time.time()}))
                os.replace(tmp, p)
            except OSError:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
                return False
            log(f"  reclaimed stale claim on shard {shard:05d}")
            return True
        finally:
            self._release_guard(guard)

    def owns(self, shard: int) -> bool:
        """True iff the claim on `shard` currently names THIS process.

        Uses the same identity rule as everything else (host + pid).  An absent or unverifiable
        claim is NOT ours -- the safe direction, because the caller uses this to decide whether it
        may commit output.
        """
        meta = self._read(self._path(shard))
        return meta is not None and self._is_mine(meta)

    def heartbeat(self, shard: int) -> bool:
        """Refresh our claim's heartbeat.

        Returns True only when the claim was ACTUALLY rewritten.  Returns False when this process
        is not (or is no longer) the owner -- a legitimate no-op, not a failure.  RAISES on a
        genuine filesystem failure, so a monitor cannot mistake an unwritten claim for a live one:
        swallowing the error here is what would let a worker believe its claim is fresh while it
        silently ages into reclaimable.
        """
        p = self._path(shard)
        meta = self._read(p)
        if meta is None or not self._is_mine(meta):
            # absent, corrupt, or someone else's: never touch another worker's claim, and never
            # adopt a claim we cannot verify
            return False
        meta = {**meta, **self.identity(), "heartbeat": time.time()}
        # per-process temp name so two workers can never collide on the same scratch path
        tmp = self.dir / f"shard_{shard:05d}.claim.hb.{os.getpid()}"
        try:
            tmp.write_text(json.dumps(meta))
            os.replace(tmp, p)
        except BaseException:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise
        return True

    def release(self, shard: int) -> None:
        """Drop our claim.  Only removes it if we still own it.

        Without the ownership check a worker that lost its claim to a reclaimer (because it was
        wedged past the heartbeat window) could delete the NEW owner's claim on its way out.
        """
        p = self._path(shard)
        meta = self._read(p)
        if meta is not None and not self._is_mine(meta):
            return
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    def sweep(self) -> int:
        """Remove orphaned private scratch files and dead reclaim guards.

        Neither blocks progress on its own -- a leaked guard is re-acquired by the age rule in
        `_acquire_guard`, and a leaked scratch file is invisible to the protocol.  This is
        tidiness, called from `bin/status.py`.
        """
        n = 0
        cutoff = time.time() - max(60, self.stale_seconds)
        for f in self.dir.glob(f"*{PRIVATE_SUFFIX}.*"):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    n += 1
            except FileNotFoundError:
                pass
        for g in self.dir.glob("*.reclaim"):
            try:
                if g.is_dir() and g.stat().st_mtime < cutoff:
                    os.rmdir(g)
                    n += 1
            except OSError:
                pass
        return n

    # ------------------------------------------------------------------ iteration
    def iter_available(self, shards, is_done, max_shards: int | None = None):
        """Yield shard ids this worker owns and that are not already done.

        `shards` is scanned in order; a worker exits cleanly when nothing is left, which is what
        lets the primary and opportunistic arrays cooperate on one directory.

        `max_shards` bounds how many shards this worker will take -- the bounded smoke mode.
        It affects only how much work ONE worker does; it never marks anything done, so a
        production run afterwards picks up everything that is still outstanding.
        """
        taken = 0
        for s in shards:
            if max_shards is not None and taken >= max_shards:
                return
            if is_done(s):
                continue
            if self.try_claim(s):
                if is_done(s):  # someone finished it between the checks
                    self.release(s)
                    continue
                taken += 1
                yield s

    def status(self, shards, is_done) -> dict:
        shards = list(shards)
        done = [s for s in shards if is_done(s)]
        done_set = set(done)
        claimed, stale = [], []
        for s in shards:
            if s in done_set:
                continue
            p = self._path(s)
            if p.exists():
                (stale if self._is_stale(p) else claimed).append(s)
        pending = [s for s in shards if s not in done_set and not self._path(s).exists()]
        return dict(
            total=len(shards),
            done=len(done),
            in_flight=len(claimed),
            stale=len(stale),
            pending=len(pending),
        )


class Heartbeat:
    """Keep a claim live while a long, blocking section runs.  Use as a context manager.

    `llm.generate()` blocks for the whole shard.  A single heartbeat before the call is not
    enough: `claim_stale_seconds` is 1800 and a 10,000-row shard is only *projected* at 15-20
    minutes, so a slow shard -- long documents, a contended node, a cold compile -- would become
    reclaimable while its owner is still actively generating, and two workers would do the same
    work.

    A daemon thread refreshes the claim every `interval` seconds until the section ends.  It is
    always joined on exit, so no thread is leaked; `daemon=True` only covers an interpreter that
    dies without unwinding.  Refreshing goes through `ClaimDir.heartbeat`, which is a no-op when
    this process is not the owner, so a worker whose claim was legitimately reclaimed never
    touches the new owner's claim.  Errors are counted and surfaced once rather than silently
    swallowed, and a failed refresh can never write a partial claim (heartbeat writes to a
    per-process temp and renames).
    """

    def __init__(self, claims: "ClaimDir", shard: int, interval: float = 300.0):
        self.claims = claims
        self.shard = shard
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.beats = 0          # refreshes that actually rewrote the claim
        self.lost = 0           # refreshes skipped because we are no longer the owner
        self.errors = 0         # refreshes that raised
        self.last_error: BaseException | None = None

    def _loop(self) -> None:
        # Event.wait doubles as the sleep AND the stop signal, so exit is immediate.
        while not self._stop.wait(self.interval):
            try:
                if self.claims.heartbeat(self.shard):
                    self.beats += 1
                else:
                    # ownership is gone; keep looping so `lost` keeps rising and the shard's
                    # ownership check refuses to commit, but stop pretending we are alive
                    self.lost += 1
            except BaseException as e:  # noqa: BLE001 -- a monitor thread must not die silently
                self.errors += 1
                self.last_error = e

    @property
    def healthy(self) -> bool:
        """No failed refresh and no lost ownership since the section began."""
        return self.errors == 0 and self.lost == 0

    def __enter__(self) -> "Heartbeat":
        if self.interval <= 0:
            return self  # disabled
        try:
            if self.claims.heartbeat(self.shard):  # refresh now, then periodically
                self.beats += 1
            else:
                self.lost += 1
        except BaseException as e:  # noqa: BLE001 -- entering must not fail the shard
            self.errors += 1
            self.last_error = e
        self._thread = threading.Thread(
            target=self._loop, name=f"heartbeat-{self.shard:05d}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Runs on the normal path AND on an exception, so the thread always stops.
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=max(5.0, min(30.0, self.interval)))
            if t.is_alive():
                log(f"  WARNING: heartbeat thread for shard {self.shard:05d} did not stop")
        if self.errors:
            log(
                f"  WARNING: {self.errors} heartbeat FAILURE(S) on shard {self.shard:05d} "
                f"({self.beats} succeeded); last: {self.last_error!r}"
            )
        if self.lost:
            log(
                f"  WARNING: lost ownership of shard {self.shard:05d} during processing "
                f"({self.lost} skipped refresh(es), {self.beats} succeeded)"
            )
        return False  # never suppress the original exception

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
