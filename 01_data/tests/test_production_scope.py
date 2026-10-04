"""Production scope (configs/production.yaml): only the allowed settings can be launched or written.

Covers the launcher's worklist and sbatch line, and the worker's own fail-closed gate on the
production output path.  The sbatch re-check is exercised against the real cluster at launch.
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import scaled_config

from kys3b.config import REWRITE_SETTINGS, Config, production_scope
from kys3b.io import Stop
from kys3b.rewrite import worker

BIN = Path(__file__).resolve().parents[1] / "bin"
IN_SCOPE = ["disagreement-aware", "quality-first", "diversity-oriented"]
EXCLUDED = ["wrap-inspired", "rewire-inspired"]


def _launcher():
    sys.path.insert(0, str(BIN))
    spec = importlib.util.spec_from_file_location("rewrite_launch", BIN / "03_rewrite_launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestScopeConfig(unittest.TestCase):
    def test_exactly_the_three_settings_in_order(self):
        s = production_scope()
        self.assertEqual(s["allowed_settings"], IN_SCOPE)
        self.assertEqual(sorted(s["excluded_settings"]), sorted(EXCLUDED))
        self.assertEqual(s["passes"], ["p1", "distill"])
        # every rewriting setting is accounted for, so a sixth cannot slip in by default
        self.assertEqual(sorted(IN_SCOPE + EXCLUDED), sorted(REWRITE_SETTINGS))


class TestLauncher(unittest.TestCase):
    def setUp(self):
        self.L = _launcher()
        self.scope = production_scope()

    def test_default_worklist_is_the_six_in_scope_items(self):
        items = self.L.worklist(self.scope, self.scope["allowed_settings"], self.scope["passes"])
        self.assertEqual(items, [f"{s}:{p}" for s in IN_SCOPE for p in ("p1", "distill")])

    def test_excluded_settings_are_refused_not_skipped(self):
        for s in EXCLUDED:
            with self.assertRaises(Stop):
                self.L.worklist(self.scope, ["quality-first", s], ["p1"])

    def test_sbatch_line_carries_only_in_scope_items_and_the_commit(self):
        cfg = Config.load()
        items = self.L.worklist(self.scope, IN_SCOPE, ["p1", "distill"])
        for q in ("primary", "opportunistic"):
            cmd = " ".join(self.L.sbatch_cmd(cfg, items, q, "deadbeef"))
            for s in EXCLUDED:
                self.assertNotIn(s, cmd)
            self.assertIn("KYS_CODE_COMMIT=deadbeef", cmd)
            self.assertIn("KYS_WORKLIST=" + "+".join(items), cmd)
            # '+' separator: a comma would be split by sbatch --export
            self.assertNotIn(",", cmd.split("KYS_WORKLIST=")[1].split(",KYS_CODE_COMMIT")[0])
        self.assertIn("--requeue", self.L.sbatch_cmd(cfg, items, "opportunistic", "x"))
        self.assertNotIn("--requeue", self.L.sbatch_cmd(cfg, items, "primary", "x"))


class TestWorkerGate(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.cfg = scaled_config(tmp_root=Path(self._d.name))
        self._p = mock.patch.object(Config, "load", staticmethod(lambda: self.cfg))
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._d.cleanup()

    def test_excluded_setting_cannot_write_production_output(self):
        for s in EXCLUDED:
            for p in ("p1", "distill"):
                with self.assertRaises(Stop):
                    worker.main(["--setting", s, "--pass", p])
                with self.assertRaises(Stop):
                    worker.main(["--setting", s, "--pass", p, "--pilot", "2",
                                 "--pilot-to-production"])
                # refused before anything is created on the production path
                self.assertFalse((self.cfg.stage("rewritten") / s / p).exists())


if __name__ == "__main__":
    unittest.main()
