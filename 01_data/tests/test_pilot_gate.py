"""The pilot gate must FAIL CLOSED: no projection from a partial pilot, ever."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
from helpers import *  # noqa: F401,F403  (sys.path setup)

from kys3b.calibration import pilot_shards, verify_pilot
from kys3b.io import atomic_write_json, atomic_write_table
from kys3b.shards import shard_path

ROOT = Path(__file__).resolve().parents[1]
K = 8
N_SHARDS = 40
ROWS = 30
SETTINGS = (
    "quality-first", "wrap-inspired", "rewire-inspired",
    "diversity-oriented", "disagreement-aware",
)


def _mk_source(src: Path, n=N_SHARDS, rows=ROWS) -> dict:
    src.mkdir(parents=True, exist_ok=True)
    shards = []
    for k in range(n):
        ids = np.arange(k * rows, (k + 1) * rows, dtype=np.int64)
        ntok = np.full(rows, 100 + k, dtype=np.int32)
        atomic_write_table(
            pa.table({
                "doc_id": pa.array(ids, type=pa.int64()),
                "text": pa.array(["x"] * rows, type=pa.large_string()),
                "tokens_llama2": pa.array(ntok, type=pa.int32()),
            }),
            shard_path(src, k),
        )
        shards.append(dict(shard=k, rows=rows, doc_id_min=int(ids[0]), doc_id_max=int(ids[-1]),
                           file=shard_path(src, k).name,
                           source_train_tokens=int(ntok.sum() + rows)))
    idx = dict(setting="t", rows_per_shard=rows, n_shards=n, total_docs=n * rows,
               total_source_train_tokens=sum(s["source_train_tokens"] for s in shards),
               shards=shards)
    atomic_write_json(idx, src / "_shards.json")
    return idx


def _mk_output(out: Path, k: int, rows=ROWS, ntok_base=100, shift=0):
    ids = np.arange(k * ROWS, k * ROWS + rows, dtype=np.int64) + shift
    atomic_write_table(
        pa.table({
            "doc_id": pa.array(ids, type=pa.int64()),
            "status": pa.array(np.full(rows, 2, np.int8), type=pa.int8()),
            "rewritten_tokens": pa.array(
                np.full(rows, (ntok_base + k) // 3, np.int32), type=pa.int32()
            ),
        }),
        shard_path(out, k),
    )


class TestVerifyPilot(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.root = Path(self._d.name)
        self.src = self.root / "src"
        self.out = self.root / "out"
        self.out.mkdir(parents=True)
        self.idx = _mk_source(self.src)
        self.expected = pilot_shards(N_SHARDS, K)

    def tearDown(self):
        self._d.cleanup()

    def _all(self):
        for k in self.expected:
            _mk_output(self.out, k)

    def test_complete_pilot_has_no_problems(self):
        self._all()
        self.assertEqual(verify_pilot(self.src, self.out, self.expected, self.idx), [])

    def test_missing_shard_is_reported(self):
        self._all()
        shard_path(self.out, self.expected[3]).unlink()
        probs = verify_pilot(self.src, self.out, self.expected, self.idx)
        self.assertEqual(len(probs), 1)
        self.assertIn("MISSING", probs[0])
        self.assertIn(f"{self.expected[3]:05d}", probs[0])

    def test_partial_row_shard_is_reported(self):
        self._all()
        _mk_output(self.out, self.expected[2], rows=7)  # partial
        probs = verify_pilot(self.src, self.out, self.expected, self.idx)
        self.assertEqual(len(probs), 1)
        self.assertIn("PARTIAL", probs[0])

    def test_misaligned_doc_ids_are_reported(self):
        self._all()
        _mk_output(self.out, self.expected[5], shift=999)
        probs = verify_pilot(self.src, self.out, self.expected, self.idx)
        self.assertEqual(len(probs), 1)
        self.assertIn("do NOT align", probs[0])

    def test_no_outputs_reports_every_shard(self):
        probs = verify_pilot(self.src, self.out, self.expected, self.idx)
        self.assertEqual(len(probs), K)

    def test_empty_expected_set_is_itself_a_problem(self):
        self.assertEqual(len(verify_pilot(self.src, self.out, [], self.idx)), 1)

    def test_extra_non_pilot_shards_are_ignored(self):
        """Only the deterministic set matters; stray outputs must not make it pass or fail."""
        self._all()
        stray = next(k for k in range(N_SHARDS) if k not in self.expected)
        _mk_output(self.out, stray)
        self.assertEqual(verify_pilot(self.src, self.out, self.expected, self.idx), [])


class TestPilotReportExitCodes(unittest.TestCase):
    """End-to-end exit codes from bin/pilot_report.py against a throwaway data root.

    Run IN-PROCESS with `kys3b.config.CONFIG_DIR` patched, rather than as a subprocess with an
    invented env var: the production config loader deliberately has no override hook, so a test
    must not pretend it does (the first attempt at this test silently read the real config and
    "passed" for the wrong reason).
    """

    @classmethod
    def setUpClass(cls):
        cls.expected = pilot_shards(N_SHARDS, K)
        if str(ROOT / "bin") not in sys.path:
            sys.path.insert(0, str(ROOT / "bin"))
        import importlib

        cls.mod = importlib.import_module("pilot_report")
        cls.cfgmod = importlib.import_module("kys3b.config")

    def setUp(self):
        import yaml

        self._d = tempfile.TemporaryDirectory()
        self.root = Path(self._d.name)
        cfgdir = self.root / "configs"
        cfgdir.mkdir(parents=True)
        for name in ("budgets.yaml", "vllm.yaml", "cluster.yaml"):
            (cfgdir / name).write_text((ROOT / "configs" / name).read_text())
        paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
        paths["roots"]["data"] = str(self.root / "data")
        paths["roots"]["pool"] = str(self.root / "data" / "00_pool")
        paths["roots"]["dataset"] = str(self.root / "data" / "dataset")
        paths["roots"]["cache"] = str(self.root / "data" / ".cache")
        (cfgdir / "paths.yaml").write_text(yaml.safe_dump(paths))
        self.ds = self.root / "data" / "dataset"
        self.idx = {s: _mk_source(self.ds / "02_sources" / s) for s in SETTINGS}
        self._saved_cfgdir = self.cfgmod.CONFIG_DIR
        self.cfgmod.CONFIG_DIR = cfgdir

    def tearDown(self):
        self.cfgmod.CONFIG_DIR = self._saved_cfgdir
        self._d.cleanup()

    def _out(self, setting, pass_name):
        d = self.ds / "03_rewritten" / setting / "_pilot" / pass_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _populate(self, settings=SETTINGS, passes=("p1", "distill"), skip=()):
        for s in settings:
            for pn in passes:
                d = self._out(s, pn)
                for k in self.expected:
                    if (s, pn, k) in skip:
                        continue
                    _mk_output(d, k)

    def _run(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.mod.main(["--shards", str(K)])
        return rc, buf.getvalue()

    def test_config_patch_actually_took_effect(self):
        """Guards the test itself: the throwaway root must be what the report reads."""
        cfg = self.cfgmod.Config.load()
        self.assertEqual(str(cfg.root("dataset")), str(self.ds))

    # 1
    def test_complete_pilot_can_pass(self):
        self._populate()
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        self.assertIn("GATE PASSED", out)

    # 2
    def test_one_missing_p1_shard_is_non_zero(self):
        self._populate(skip={("quality-first", "p1", self.expected[4])})
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertNotIn("GATE PASSED", out)
        self.assertIn("MISSING", out)
        self.assertIn("quality-first", out)

    # 3
    def test_one_missing_distill_shard_is_non_zero(self):
        self._populate(skip={("rewire-inspired", "distill", self.expected[0])})
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertNotIn("GATE PASSED", out)
        self.assertIn("rewire-inspired", out)

    # 4
    def test_only_one_pass_complete_is_a_blocker(self):
        self._populate(passes=("p1",))
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertNotIn("GATE PASSED", out)
        self.assertIn("BLOCKER", out)

    # 5
    def test_zero_pilot_outputs_is_non_zero(self):
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertNotIn("GATE PASSED", out)

    # 6
    def test_partial_row_shard_is_non_zero(self):
        self._populate()
        _mk_output(self._out("wrap-inspired", "p1"), self.expected[1], rows=5)
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertNotIn("GATE PASSED", out)
        self.assertIn("PARTIAL", out)

    def test_misaligned_shard_is_non_zero(self):
        self._populate()
        _mk_output(self._out("diversity-oriented", "distill"), self.expected[2], shift=5000)
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertIn("do NOT align", out)

    def test_no_production_go_recommendation_when_blocked(self):
        self._populate(skip={("quality-first", "p1", self.expected[4])})
        rc, out = self._run()
        self.assertNotEqual(rc, 0)
        self.assertNotIn("GATE PASSED", out)
        self.assertIn("No production-go recommendation", out)
        rep = json.loads((self.ds / "reports" / "pilot_pilot.json").read_text())
        self.assertFalse(rep["complete"])
        self.assertIn("quality-first", rep["blocking"])

    def test_a_low_headroom_arm_fails_the_gate_even_when_complete(self):
        """Completeness is necessary, not sufficient: the yield must also clear the target."""
        self._populate()
        # rewrite quality-first outputs with a tiny yield
        for pn in ("p1", "distill"):
            d = self._out("quality-first", pn)
            for k in self.expected:
                ids = np.arange(k * ROWS, (k + 1) * ROWS, dtype=np.int64)
                atomic_write_table(
                    pa.table({
                        "doc_id": pa.array(ids, type=pa.int64()),
                        "status": pa.array(np.full(ROWS, 2, np.int8), type=pa.int8()),
                        "rewritten_tokens": pa.array(np.full(ROWS, 2, np.int32), type=pa.int32()),
                    }),
                    shard_path(d, k),
                )
        rc, out = self._run()
        self.assertEqual(rc, 1, out)
        self.assertNotIn("GATE PASSED", out)
        self.assertIn("BELOW TARGET", out)


if __name__ == "__main__":
    unittest.main()
