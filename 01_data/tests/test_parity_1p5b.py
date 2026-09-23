"""Differential parity against the ACTUAL 1.5B implementation, where it is still on disk.

This is the strongest available check that the ported rules are behaviourally identical: the
1.5B modules are loaded directly and run side by side against the 3B ports.  Skipped (not
failed) if `projects/rewrite/` is absent, so the suite still runs elsewhere.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
from helpers import *  # noqa: F401,F403

ONE5B = Path("/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite")
PP_IO = ONE5B / "10_postprocess" / "pp_io.py"
WRAP_STRIP = ONE5B / "10_postprocess" / "01_strip_prefix_wrap.py"
SELECT_10B = ONE5B / "04_select" / "select_10b.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DISTILL_CASES = [
    "### Paraphrased Text\n\nbody",
    "Here is a condensed version:\n\nbody",
    "Paraphrased Text:\n\nbody",
    "**Paraphrased Version:**\nbody",
    "### Paraphrased Version:\nbody",
    "Arsenic contamination in drinking water has been a problem.\n\nmore",
    "### Frequently Asked Questions\n\nQ: x",
    "Sure, here is the rewritten summary:\n\nbody",
    "Certainly! Here is your condensed version:\nbody",
    "Here is a paraphrased version of the text " + "x" * 120 + "\n\nbody",
    "## Condensed Version\n\nbody",
    "The following is a rewritten summary:\n\nbody",
    "",
    "no newlines at all",
    "Q: what?\nA: this.",
]
WRAP_CASES = DISTILL_CASES + [
    "Here is the rewritten passage:\n\nBody.",
    "Here is Paris, a city in France.\n\nBody.",
    "### Simple Version for Young Children\n\nBody.",
    "### Case Summary\n\nreal content",
    "### Passage\n\nreal content",
    "Rewritten passage:\n\nBody.",
    "I have rewritten the passage in plain language.\n\nBody.",
    "Of course! Here is the simplified version:\n\nBody.",
    "**Scholarly Language Version**\n\nBody.",
    "Here is the rewritten passage " + "y" * 300 + "\n\nbody",
]
LEAK_CASES = [
    "Important: Do not add any information, claims, or details that are not stated.\n\nreal",
    "Important: Do not add any information, claims, or details that are not stated.",
    "normal text",
    "",
]


@unittest.skipUnless(PP_IO.exists(), "the 1.5B repository is not on this machine")
class TestStripParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pp = _load(PP_IO, "pp_io_1p5b")
        from kys3b.post import strip as s3

        cls.s3 = s3

    def test_strip_distill_preamble_identical(self):
        for c in DISTILL_CASES:
            self.assertEqual(
                self.pp.strip_distill_preamble(c),
                self.s3.strip_distill_preamble(c),
                f"distill rule differs on {c[:60]!r}",
            )

    def test_strip_instruction_leak_identical(self):
        for c in LEAK_CASES:
            self.assertEqual(
                self.pp.strip_instruction_leak(c),
                self.s3.strip_instruction_leak(c),
                f"leak rule differs on {c[:60]!r}",
            )

    def test_constants_identical(self):
        self.assertEqual(
            self.pp.DISTILL_MAX_PREAMBLE_CHARS, self.s3.DISTILL_MAX_PREAMBLE_CHARS
        )
        self.assertEqual(self.pp.DISTILL_PREAMBLE_WORDS, self.s3.DISTILL_PREAMBLE_WORDS)
        self.assertEqual(self.pp.DISTILL_OPENERS, self.s3.DISTILL_OPENERS)
        self.assertEqual(self.pp.INSTRUCTION_LEAK_ANCHOR, self.s3.INSTRUCTION_LEAK_ANCHOR)


@unittest.skipUnless(WRAP_STRIP.exists(), "the 1.5B repository is not on this machine")
class TestWrapStripParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = WRAP_STRIP.read_text()
        # the 1.5B script is not importable (module-level paths + a `main`), so lift the rule
        # and its constants out of the source and exec just those
        start = src.index("\nMAX_PREAMBLE_CHARS = ") + 1
        end = src.index("# --------------------------------------------------------"
                        "--------------------- worker tokenizer")
        ns: dict = {"os": __import__("os"), "sys": __import__("sys"),
                    "time": __import__("time"), "re": __import__("re")}
        exec(compile(src[start:end], "wrap_rule_1p5b", "exec"), ns)
        cls.ref = ns
        from kys3b.post import strip as s3

        cls.s3 = s3

    def test_constants_identical(self):
        self.assertEqual(self.ref["MAX_PREAMBLE_CHARS"], self.s3.WRAP_MAX_PREAMBLE_CHARS)
        self.assertEqual(self.ref["OPENERS"], self.s3.WRAP_OPENERS)
        self.assertEqual(self.ref["SIGNAL_WORDS"], self.s3.WRAP_SIGNAL_WORDS)
        self.assertEqual(self.ref["STRICT_META"], self.s3.WRAP_STRICT_META)

    def test_strip_preamble_identical(self):
        for c in WRAP_CASES:
            self.assertEqual(
                self.ref["strip_preamble"](c),
                self.s3.strip_wrap_preamble(c),
                f"wrap rule differs on {c[:60]!r}",
            )


@unittest.skipUnless(SELECT_10B.exists(), "the 1.5B repository is not on this machine")
class TestSelectionPrimitiveParity(unittest.TestCase):
    """`fill_to` and `order_desc` must behave exactly as the 1.5B originals."""

    @classmethod
    def setUpClass(cls):
        src = SELECT_10B.read_text()
        start = src.index("def order_desc")
        end = src.index("def pctl(")
        ns: dict = {"np": np}
        exec(compile(src[start:end], "select_prims_1p5b", "exec"), ns)
        cls.ref = ns

    def test_fill_to_identical_on_random_inputs(self):
        from kys3b.select import fill_to

        rng = np.random.default_rng(3)
        for _ in range(300):
            n = int(rng.integers(1, 200))
            tok = rng.integers(1, 5000, size=n).astype(np.int64)
            order = rng.permutation(n)
            target = float(rng.integers(1, int(tok.sum() * 1.3) + 2))
            a = self.ref["fill_to"](order, tok, target)
            b = fill_to(order, tok, target)
            self.assertTrue(np.array_equal(a[0], b[0]), "selected sets differ")
            self.assertEqual(a[1:], b[1:], "totals/overshoot/filled differ")

    def test_order_desc_identical_with_ties(self):
        from kys3b.select import order_desc

        rng = np.random.default_rng(4)
        for _ in range(100):
            n = int(rng.integers(2, 500))
            score = np.round(rng.random(n), 2).astype(np.float32)  # lots of ties
            tie = rng.permutation(n).astype(np.int64)
            idxs = rng.permutation(n)[: max(1, n // 2)]
            self.assertTrue(
                np.array_equal(
                    self.ref["order_desc"](idxs, score, tie), order_desc(idxs, score, tie)
                )
            )

    def test_qv_formula_identical(self):
        """q and v must match select_10b.py:198-199 exactly, in float32 with ddof=0."""
        from kys3b.disagreement import qv_from_columns

        rng = np.random.default_rng(5)
        ft = rng.random(1000).astype(np.float32)
        fw = rng.random(1000).astype(np.float32)
        mb = rng.random(1000).astype(np.float32)
        q_ref = ((ft + fw + mb) / 3.0).astype(np.float32)
        v_ref = (((ft - q_ref) ** 2 + (fw - q_ref) ** 2 + (mb - q_ref) ** 2) / 3.0).astype(
            np.float32
        )
        q, v = qv_from_columns(ft, fw, mb)
        self.assertTrue(np.array_equal(q, q_ref))
        self.assertTrue(np.array_equal(v, v_ref))


if __name__ == "__main__":
    unittest.main()
