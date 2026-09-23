"""Worker identity, cache namespaces and progress paths must be unique ACROSS the two arrays.

The primary jhu2 array and the opportunistic scavenger array both number their tasks from 0, so
a namespace keyed on the task id alone would let primary task 0 and scavenger task 0 share
TMPDIR / VLLM_CACHE_ROOT / TORCHINDUCTOR_CACHE_DIR and overwrite each other's progress JSON
while running concurrently.
"""
import os
import re
import subprocess
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403  (sys.path setup)

from kys3b.claims import ClaimDir

SLURM_VARS = ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID")
SBATCH = Path(__file__).resolve().parents[1] / "slurm" / "rewrite_array.sbatch"


class _SlurmEnv:
    def __init__(self, **kw):
        self.kw = kw
        self.saved: dict = {}

    def __enter__(self):
        for k in SLURM_VARS:
            self.saved[k] = os.environ.get(k)
            os.environ.pop(k, None)
        for k, v in self.kw.items():
            os.environ[k] = str(v)
        return self

    def __exit__(self, *a):
        for k in SLURM_VARS:
            os.environ.pop(k, None)
            if self.saved[k] is not None:
                os.environ[k] = self.saved[k]
        return False


def primary(task):
    return _SlurmEnv(SLURM_ARRAY_JOB_ID=111111, SLURM_ARRAY_TASK_ID=task,
                     SLURM_JOB_ID=111111 + task)


def scavenger(task):
    return _SlurmEnv(SLURM_ARRAY_JOB_ID=222222, SLURM_ARRAY_TASK_ID=task,
                     SLURM_JOB_ID=222222 + task)


class TestWorkerKeyUniqueness(unittest.TestCase):
    def test_same_task_id_different_array_gives_different_worker_key(self):
        with primary(0):
            a = ClaimDir.worker_key()
        with scavenger(0):
            b = ClaimDir.worker_key()
        self.assertNotEqual(a, b, "primary task 0 and scavenger task 0 must differ")
        self.assertIn("111111", a)
        self.assertIn("222222", b)
        self.assertIn("_t0_", a)
        self.assertIn("_t0_", b)

    def test_same_array_different_task_gives_different_worker_key(self):
        with primary(0):
            a = ClaimDir.worker_key()
        with primary(7):
            b = ClaimDir.worker_key()
        self.assertNotEqual(a, b)

    def test_identity_records_the_array_job(self):
        with scavenger(5):
            i = ClaimDir.identity()
        self.assertEqual(i["array_job"], "222222")
        self.assertEqual(i["worker"], "5")
        self.assertEqual(i["jobid"], "222227")

    def test_array_job_falls_back_to_job_id_then_local(self):
        with _SlurmEnv(SLURM_JOB_ID=98765):
            self.assertEqual(ClaimDir.identity()["array_job"], "98765")
        with _SlurmEnv():
            i = ClaimDir.identity()
            self.assertEqual(i["array_job"], "local")
            self.assertEqual(i["worker"], "local")

    def test_worker_key_is_filename_safe(self):
        with primary(3):
            k = ClaimDir.worker_key()
        self.assertRegex(k, r"^[A-Za-z0-9._\-]+$", f"not filename-safe: {k}")
        self.assertEqual(Path(f"worker_{k}.json").name, f"worker_{k}.json")

    def test_progress_paths_do_not_collide_across_arrays(self):
        names = set()
        for ctx in (primary(0), scavenger(0), primary(1), scavenger(1)):
            with ctx:
                names.add(f"worker_{ClaimDir.worker_key()}.json")
        self.assertEqual(len(names), 4, f"progress filenames collided: {names}")


class TestCacheNamespace(unittest.TestCase):
    """Evaluate the sbatch's own KYS_NS expression, so the test tracks the real script."""

    @staticmethod
    def _ns_expr():
        m = re.search(r'^export KYS_NS="(rewrite/[^"]+)"$', SBATCH.read_text(), re.M)
        assert m, "could not find the KYS_NS assignment in rewrite_array.sbatch"
        return m.group(1)

    def _resolve(self, array_job, task, setting="quality-first", pass_name="p1"):
        env = dict(os.environ)
        env.update(
            KYS_SETTING=setting, KYS_PASS=pass_name,
            SLURM_ARRAY_JOB_ID=str(array_job), SLURM_ARRAY_TASK_ID=str(task),
            SLURM_JOB_ID=str(array_job + task),
        )
        out = subprocess.run(
            ["bash", "-c", f'printf "%s" "{self._ns_expr()}"'],
            env=env, capture_output=True, text=True, check=True,
        )
        return out.stdout

    def test_namespace_includes_the_array_job_id(self):
        a = self._resolve(111111, 0)
        b = self._resolve(222222, 0)
        self.assertNotEqual(a, b, f"namespaces collide across arrays: {a} == {b}")
        self.assertIn("111111", a)
        self.assertIn("222222", b)

    def test_namespace_distinguishes_tasks_settings_and_passes(self):
        seen = {
            self._resolve(111111, 0),
            self._resolve(111111, 1),
            self._resolve(222222, 0),
            self._resolve(111111, 0, setting="wrap-inspired"),
            self._resolve(111111, 0, pass_name="distill"),
        }
        self.assertEqual(len(seen), 5, f"namespaces collided: {seen}")

    def test_derived_cache_dirs_are_disjoint(self):
        """TMPDIR / VLLM_CACHE_ROOT / TORCHINDUCTOR_CACHE_DIR all hang off KYS_NS."""
        a, b = self._resolve(111111, 0), self._resolve(222222, 0)
        for base in ("tmp", "vllm", "torchinductor"):
            self.assertNotEqual(f"{base}/{a}", f"{base}/{b}")

    def test_falls_back_cleanly_outside_slurm(self):
        env = dict(os.environ)
        for k in SLURM_VARS:
            env.pop(k, None)
        env.update(KYS_SETTING="quality-first", KYS_PASS="p1")
        out = subprocess.run(
            ["bash", "-c", f'printf "%s" "{self._ns_expr()}"'],
            env=env, capture_output=True, text=True, check=True,
        )
        self.assertEqual(out.stdout, "rewrite/quality-first/p1/local/0")


if __name__ == "__main__":
    unittest.main()
