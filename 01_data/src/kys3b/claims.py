"""Claim-directory work distribution with stale-claim recovery (Decision 8).

Replaces the 1.5B `shard_idx % num_workers` assignment.  Modulo assignment strands a dead
worker's shards until that exact array index is resubmitted, which is unacceptable on a
preemptible scavenger queue.  This is BEHAVIOURALLY NEUTRAL: per-document output depends
only on (shard_index, row_index), never on which worker ran the shard.

A claim is an exclusively-created file holding JSON {worker, host, jobid, pid, heartbeat}.
A claim is reclaimable when
  * its heartbeat is older than `stale_seconds`, OR
  * its Slurm job id is no longer in `squeue` (checked lazily, cached per process).

Completion is recorded by the OUTPUT parquet existing, never by the claim: an output that
exists is never redone, which is what makes a rerun idempotent.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from .io import log


class ClaimDir:
    def __init__(self, directory: Path, stale_seconds: int = 1800):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stale_seconds = int(stale_seconds)
        self._live_jobs: set[str] | None = None
        self._live_jobs_at = 0.0

    # ------------------------------------------------------------------ identity
    @staticmethod
    def identity() -> dict:
        return dict(
            worker=os.environ.get("SLURM_ARRAY_TASK_ID", "local"),
            host=socket.gethostname(),
            jobid=os.environ.get("SLURM_JOB_ID", ""),
            pid=os.getpid(),
        )

    def _path(self, shard: int) -> Path:
        return self.dir / f"shard_{shard:05d}.claim"

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
            else:
                jobs = set()
        except Exception:
            jobs = set()
        self._live_jobs, self._live_jobs_at = jobs, time.time()
        return jobs

    def _is_stale(self, p: Path) -> bool:
        try:
            m = json.loads(p.read_text())
        except Exception:
            # unreadable/truncated claim: treat as stale
            return True
        age = time.time() - float(m.get("heartbeat", 0))
        if age > self.stale_seconds:
            return True
        jobid = str(m.get("jobid") or "")
        if jobid:
            alive = self._jobs_alive()
            # only trust squeue when it returned something; an empty set may mean it failed
            if alive and jobid not in alive:
                return True
        return False

    # ------------------------------------------------------------------ claim / release
    def try_claim(self, shard: int) -> bool:
        p = self._path(shard)
        payload = json.dumps({**self.identity(), "heartbeat": time.time()})
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            return True
        except FileExistsError:
            if self._is_stale(p):
                # steal it: rewrite in place, then verify we are the owner
                try:
                    tmp = p.with_suffix(".claim.steal")
                    tmp.write_text(payload)
                    os.replace(tmp, p)
                    time.sleep(0.05)
                    cur = json.loads(p.read_text())
                    if cur.get("pid") == os.getpid() and cur.get("host") == socket.gethostname():
                        log(f"  reclaimed stale claim on shard {shard:05d}")
                        return True
                except Exception:
                    return False
            return False

    def heartbeat(self, shard: int) -> None:
        p = self._path(shard)
        try:
            m = json.loads(p.read_text())
        except Exception:
            m = self.identity()
        m["heartbeat"] = time.time()
        try:
            tmp = p.with_suffix(".claim.hb")
            tmp.write_text(json.dumps(m))
            os.replace(tmp, p)
        except Exception:
            pass

    def release(self, shard: int) -> None:
        try:
            self._path(shard).unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------ iteration
    def iter_available(self, shards, is_done) -> "iter":
        """Yield shard ids this worker successfully claimed and that are not already done.

        `shards` is scanned in order; a worker exits cleanly when nothing is left, which is
        what lets the primary and opportunistic arrays cooperate on one directory.
        """
        for s in shards:
            if is_done(s):
                continue
            if self.try_claim(s):
                if is_done(s):  # someone finished it between the checks
                    self.release(s)
                    continue
                yield s

    def status(self, shards, is_done) -> dict:
        done = [s for s in shards if is_done(s)]
        claimed, stale = [], []
        for s in shards:
            if s in done:
                continue
            p = self._path(s)
            if p.exists():
                (stale if self._is_stale(p) else claimed).append(s)
        pending = [s for s in shards if s not in done and not self._path(s).exists()]
        return dict(
            total=len(list(shards)),
            done=len(done),
            in_flight=len(claimed),
            stale=len(stale),
            pending=len(pending),
        )
