"""Prompt provenance (md5) and the templated-overhead gate against the real Qwen tokenizer."""
import unittest

from helpers import *  # noqa: F401,F403

from kys3b.config import WRAP_STYLES, Config
from kys3b.io import Stop
from kys3b.prompts import (
    MD5,
    build_content,
    check_md5,
    check_overheads,
    load_text,
    wrap_prompts,
)


class TestPromptProvenance(unittest.TestCase):
    def test_md5s_match_the_1p5b_production_set(self):
        got = check_md5()
        self.assertEqual(got, MD5)

    def test_expected_md5_values_are_the_documented_ones(self):
        self.assertEqual(MD5["p1_wiki"], "bca104fe6e298615e5ccb9c9c747073b")
        self.assertEqual(MD5["p2_distill"], "538700534e99d5e80b268fd9b2408b48")
        self.assertEqual(MD5["wrap_easy"], "0735f53aca80cadaa8d67727680dbbfd")
        self.assertEqual(MD5["wrap_hard"], "e99a613bcd4146416428d576af6f200a")
        self.assertEqual(MD5["wrap_wiki"], "cec46736de0229e6d7a0f022cd2e661a")
        self.assertEqual(MD5["wrap_qa"], "733fbeea43050cb4a4e27f9384b9014e")

    def test_abandoned_paper_verbatim_wrap_set_is_not_present(self):
        """The abandoned set has a 'medium' style; the production set must not."""
        self.assertNotIn("medium", wrap_prompts())

    def test_grounded_substitutes_the_placeholder(self):
        t = load_text("p1_wiki")
        self.assertIn("[TEXT]", t)
        out = build_content("grounded", "HELLO DOC", t, None, None)
        self.assertIn("HELLO DOC", out)
        self.assertNotIn("[TEXT]", out)

    def test_distill_placeholder_is_mid_template_and_still_substituted(self):
        t = load_text("p2_distill")
        self.assertIn("[TEXT]", t)
        self.assertLess(t.index("[TEXT]"), len(t) - 20, "the distill placeholder is mid-template")
        out = build_content("grounded", "ZZZ", t, None, None)
        self.assertIn("ZZZ", out)
        self.assertNotIn("[TEXT]", out)

    def test_wrap_appends_the_document_after_Passage(self):
        wp = wrap_prompts()
        for s in WRAP_STYLES:
            self.assertTrue(wp[s].endswith("Passage:\n"), s)
            out = build_content("wrap", "BODY", None, wp, s)
            self.assertTrue(out.endswith("Passage:\nBODY"), s)

    def test_grounded_prompt_carries_the_added_grounding_instruction(self):
        t = load_text("p1_wiki")
        self.assertIn("Do not add any information", t)


class TestPromptOverheads(unittest.TestCase):
    """The real gate: templated empty-document length must be 150/185/72/66/73/83."""

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer

            cls.cfg = Config.load()
            cls.tok = AutoTokenizer.from_pretrained(str(cls.cfg.model("qwen")), use_fast=True)
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"Qwen tokenizer unavailable: {e}")

    def test_all_six_overheads(self):
        got = check_overheads(self.tok, self.cfg)
        self.assertEqual(
            got,
            dict(p1_wiki=150, p2_distill=185, wrap_easy=72, wrap_hard=66, wrap_wiki=73, wrap_qa=83),
        )

    def test_a_changed_expectation_is_a_hard_stop(self):
        cfg = Config.load()
        cfg.vllm["prompt_overheads"]["p1_wiki"] = 149
        with self.assertRaises(Stop):
            check_overheads(self.tok, cfg, modes=("grounded_p1",))

    def test_wrap_asserts_four_values_not_one(self):
        cfg = Config.load()
        cfg.vllm["prompt_overheads"]["wrap_qa"] = 82  # only the LAST style is wrong
        with self.assertRaises(Stop):
            check_overheads(self.tok, cfg, modes=("wrap",))


if __name__ == "__main__":
    unittest.main()
