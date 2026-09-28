"""End-to-end (CPU) check that a --smoke-rows run touches NO production state.

Runs the real worker against a synthetic temporary dataset with a stub `vllm` engine (no GPU) and
the real Qwen / Llama-2 tokenizers, then runs the real `bin/status.py` on the result.
"""
import contextlib
import io
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from helpers import scaled_config

from kys3b.config import Config
from kys3b.io import atomic_write_json, atomic_write_table, parquet_rows
from kys3b.rewrite import worker
from kys3b.shards import shard_path

BIN = Path(__file__).resolve().parents[1] / "bin"
SETTING = "quality-first"
ROWS, N_SHARDS, SMOKE = 10, 3, 4


def _stub_vllm():
    m = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    class LLM:
        def __init__(self, model, **kw):
            pass

        def generate(self, prompts, params):
            out = []
            for i, _ in enumerate(prompts):
                o = types.SimpleNamespace(text=f"stub rewrite {i}", finish_reason="stop")
                out.append(types.SimpleNamespace(outputs=[o]))
            return out

    m.LLM, m.SamplingParams = LLM, SamplingParams
    return m


class TestSmokeIsolationEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg0 = Config.load()
        if not (cfg0.model("qwen").exists() and cfg0.model("llama2").exists()):
            raise unittest.SkipTest("real tokenizers not available on this host")
        # Import the tokenizer stack BEFORE any test patches sys.modules: patch.dict removes
        # modules first imported inside it, and torch cannot be re-imported in one process.
        from transformers import AutoTokenizer  # noqa: F401

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.cfg = scaled_config(tmp_root=Path(self._d.name))
        src = self.cfg.stage("sources") / SETTING
        shards = []
        for k in range(N_SHARDS):
            ids = np.arange(k * ROWS, (k + 1) * ROWS, dtype=np.int64) * 7
            atomic_write_table(pa.table({
                "doc_id": pa.array(ids, type=pa.int64()),
                "text": pa.array([f"Document {d}. The river floods each spring." for d in ids],
                                 type=pa.large_string()),
                "tokens_llama2": pa.array(np.full(ROWS, 12, np.int32), type=pa.int32()),
            }), shard_path(src, k))
            shards.append(dict(shard=k, rows=ROWS, doc_id_min=int(ids[0]),
                               doc_id_max=int(ids[-1]), file=shard_path(src, k).name,
                               source_train_tokens=13 * ROWS))
        atomic_write_json(dict(setting=SETTING, rows_per_shard=ROWS, n_shards=N_SHARDS,
                               total_docs=N_SHARDS * ROWS, shards=shards), src / "_shards.json")
        self.base = self.cfg.stage("rewritten") / SETTING
        self._patches = [
            mock.patch.object(Config, "load", staticmethod(lambda: self.cfg)),
            mock.patch.dict(sys.modules, {"vllm": _stub_vllm()}),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._d.cleanup()

    def _worker(self, pass_name, *extra):
        self.assertEqual(worker.main(["--setting", SETTING, "--pass", pass_name, *extra]), 0)

    def _status(self, pass_name):
        """(done, out_tok) for SETTING/pass from the real bin/status.py."""
        buf = io.StringIO()
        sys.path.insert(0, str(BIN))
        try:
            with contextlib.redirect_stdout(buf), mock.patch.object(
                sys, "argv", ["status.py", "--settings", SETTING]
            ):
                with self.assertRaises(SystemExit) as cm:
                    runpy.run_path(str(BIN / "status.py"), run_name="__main__")
            self.assertEqual(cm.exception.code, 0)
        finally:
            sys.path.remove(str(BIN))
        for line in buf.getvalue().splitlines():
            f = line.split()
            if f[:2] == [SETTING, pass_name]:
                return int(f[3].replace(",", "")), int(f[7].replace(",", ""))
        self.fail(f"no status line for {pass_name}:\n{buf.getvalue()}")

    def _files(self, d):
        return sorted(p.name for p in Path(d).rglob("*") if p.is_file()) if Path(d).exists() else []

    def test_smoke_state_is_isolated_and_production_shard0_stays_outstanding(self):
        for pass_name in ("p1", "distill"):
            self._worker(pass_name, "--max-shards", "1", "--smoke-rows", str(SMOKE))
            sm = self.base / "_smoke"
            t = pq.read_table(shard_path(sm / pass_name, 0))
            self.assertEqual(t.num_rows, SMOKE)
            self.assertEqual(t.column("doc_id").to_pylist(), [d * 7 for d in range(SMOKE)])
            # smoke state lives only under _smoke/
            self.assertEqual(len(list((sm / "_progress" / pass_name).glob("worker_*.json"))), 1)
            self.assertEqual(self._files(sm / "_claims" / pass_name), [], "claim not released")
            # production state untouched
            for prod in (self.base / pass_name, self.base / "_claims" / pass_name,
                         self.base / "_progress" / pass_name):
                self.assertEqual(self._files(prod), [], f"smoke wrote into {prod}")
            done, tok = self._status(pass_name)
            self.assertEqual((done, tok), (0, 0), "status must ignore smoke shards and tokens")

        # a second smoke run skips its own finished shard 0 and takes shard 1
        self._worker("p1", "--max-shards", "1", "--smoke-rows", str(SMOKE))
        self.assertEqual(parquet_rows(shard_path(self.base / "_smoke" / "p1", 1)), SMOKE)

        # production still sees shard 0 as outstanding and processes it in full
        self._worker("p1", "--max-shards", "1")
        self.assertEqual(parquet_rows(shard_path(self.base / "p1", 0)), ROWS)
        prog = [json.loads(p.read_text())
                for p in (self.base / "_progress" / "p1").glob("worker_*.json")]
        self.assertEqual(len(prog), 1)
        done, tok = self._status("p1")
        self.assertEqual(done, 1)
        self.assertEqual(tok, prog[0]["total_output_tokens"],
                         "production total must equal production progress only")

    def test_pilot_state_is_isolated_and_production_stays_outstanding(self):
        ids = list(range(N_SHARDS))  # pilot_shards(3, 12) is every shard
        for pass_name in ("p1", "distill"):
            self._worker(pass_name, "--pilot", "12")
            pl = self.base / "_pilot"
            for k in ids:
                self.assertEqual(parquet_rows(shard_path(pl / pass_name, k)), ROWS)
            self.assertEqual(len(list((pl / "_progress" / pass_name).glob("worker_*.json"))), 1)
            self.assertEqual(self._files(self.base / "_claims_pilot" / pass_name), [],
                             "claim not released")
            for prod in (self.base / pass_name, self.base / "_claims" / pass_name,
                         self.base / "_progress" / pass_name):
                self.assertEqual(self._files(prod), [], f"pilot wrote into {prod}")
            self.assertFalse((self.cfg.stage("logs") / f"monitor_{SETTING}_{pass_name}.md")
                             .exists(), "pilot wrote the production monitor file")
            self.assertEqual(self._status(pass_name), (0, 0),
                             "status must ignore pilot shards and tokens")

        # a rerun of the pilot is a no-op; production still processes every shard
        self._worker("p1", "--pilot", "12")
        self.assertEqual(len(list((self.base / "_pilot" / "_progress" / "p1")
                                  .glob("worker_*.json"))), 1)
        self._worker("p1")
        self.assertEqual(self._status("p1")[0], N_SHARDS)

    def test_production_and_pilot_paths_are_unchanged(self):
        b, logs = self.base, self.cfg.stage("logs")
        for pn in ("p1", "distill"):
            self.assertEqual(worker.run_paths(self.cfg, SETTING, pn), dict(
                out_dir=b / pn, done_dir=b / pn, claims_dir=b / "_claims" / pn,
                prog_dir=b / "_progress" / pn,
                monitor_file=logs / f"monitor_{SETTING}_{pn}.md"))
            self.assertEqual(worker.run_paths(self.cfg, SETTING, pn, pilot_k=12), dict(
                out_dir=b / "_pilot" / pn, done_dir=b / "_pilot" / pn,
                claims_dir=b / "_claims_pilot" / pn, prog_dir=b / "_pilot" / "_progress" / pn,
                monitor_file=logs / "pilot" / f"monitor_{SETTING}_{pn}.md"))
            self.assertEqual(worker.run_paths(self.cfg, SETTING, pn, pilot_k=12,
                                              pilot_to_prod=True),
                             worker.run_paths(self.cfg, SETTING, pn))
        self.assertEqual(worker.expected_rows(10_000), 10_000)
        self.assertEqual(worker.expected_rows(10_000, 64), 64)
        self.assertEqual(worker.expected_rows(3, 64), 3)


if __name__ == "__main__":
    unittest.main()
