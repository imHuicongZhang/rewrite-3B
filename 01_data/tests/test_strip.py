"""The preamble-strip rules, ported verbatim from the 1.5B pipeline.

The protection cases matter as much as the strip cases: a rule that ate
"### Frequently Asked Questions" or a qa "Q:" opening would silently damage the corpus.
"""
import unittest

from helpers import *  # noqa: F401,F403

from kys3b.post.strip import (
    WIKI_PREFIX,
    rule_for,
    strip_distill_preamble,
    strip_instruction_leak,
    strip_wiki,
    strip_wrap_preamble,
)


class TestWikiStrip(unittest.TestCase):
    def test_exact_prefix_start_anchored(self):
        body = "Paris is the capital of France."
        out, did = strip_wiki(WIKI_PREFIX + body)
        self.assertTrue(did)
        self.assertEqual(out, body)

    def test_mid_text_occurrence_is_never_touched(self):
        t = "Some prose. Here is a paraphrased version:\n\nmore prose."
        out, did = strip_wiki(t)
        self.assertFalse(did)
        self.assertEqual(out, t, "must never use .replace()")

    def test_case_sensitive(self):
        t = "here is a paraphrased version:\n\nbody"
        out, did = strip_wiki(t)
        self.assertFalse(did)
        self.assertEqual(out, t)

    def test_instruction_leak_block(self):
        leak = (
            "Important: Do not add any information, claims, or details that are not "
            "explicitly stated.\n\nThe real article begins here."
        )
        out, did = strip_instruction_leak(leak)
        self.assertTrue(did)
        self.assertEqual(out, "The real article begins here.")

    def test_whole_document_is_the_leak(self):
        out, did = strip_instruction_leak(
            "Important: Do not add any information, claims, or details that are not stated."
        )
        self.assertTrue(did)
        self.assertEqual(out, "")

    def test_idempotent(self):
        once, _ = strip_wiki(WIKI_PREFIX + "body text")
        twice, did = strip_wiki(once)
        self.assertFalse(did)
        self.assertEqual(once, twice)


class TestDistillStrip(unittest.TestCase):
    def test_markdown_header_with_meta_word(self):
        out, did = strip_distill_preamble("### Paraphrased Text\n\nThe body.")
        self.assertTrue(did)
        self.assertEqual(out, "The body.")

    def test_opener_with_meta_word(self):
        out, did = strip_distill_preamble("Here is a condensed version:\n\nThe body.")
        self.assertTrue(did)
        self.assertEqual(out, "The body.")

    def test_bare_paraphrased_label(self):
        out, did = strip_distill_preamble("Paraphrased Text:\n\nThe body.")
        self.assertTrue(did)
        self.assertEqual(out, "The body.")

    def test_first_line_rule_single_newline(self):
        out, did = strip_distill_preamble("### Paraphrased Version:\nThe body.")
        self.assertTrue(did)
        self.assertEqual(out, "The body.")

    def test_first_line_rule_requires_the_head_to_END_with_a_colon(self):
        """Faithful to 1.5B: branch (b) gates on head.endswith(':').

        `**Paraphrased Version:**` ends with '**', so the 1.5B rule does NOT strip it under the
        first-line branch.  Preserved deliberately -- this is what produced the published corpus.
        """
        t = "**Paraphrased Version:**\nThe body."
        out, did = strip_distill_preamble(t)
        self.assertFalse(did)
        self.assertEqual(out, t)

    def test_content_opening_untouched(self):
        t = "Arsenic contamination in drinking water has been a problem.\n\nMore text."
        out, did = strip_distill_preamble(t)
        self.assertFalse(did)
        self.assertEqual(out, t)

    def test_preamble_longer_than_120_chars_untouched(self):
        head = "Here is a paraphrased version of the text " + "x" * 120
        t = head + "\n\nbody"
        out, did = strip_distill_preamble(t)
        self.assertFalse(did, "the 120-char cap must hold")
        self.assertEqual(out, t)

    def test_genuine_content_header_without_meta_word_untouched(self):
        t = "### Frequently Asked Questions\n\nQ: what?\nA: this."
        out, did = strip_distill_preamble(t)
        self.assertFalse(did)


class TestWrapStrip(unittest.TestCase):
    def test_sentence_opener_needs_both_opener_and_signal_word(self):
        out, did = strip_wrap_preamble("Here is the rewritten passage:\n\nBody.")
        self.assertTrue(did)
        self.assertEqual(out, "Body.")
        # an opener with NO signal word is left alone
        out2, did2 = strip_wrap_preamble("Here is Paris, a city in France.\n\nBody.")
        self.assertFalse(did2)

    def test_markdown_header_gated_on_strict_meta(self):
        out, did = strip_wrap_preamble("### Simple Version for Young Children\n\nBody.")
        self.assertTrue(did)
        self.assertEqual(out, "Body.")

    def test_genuine_content_headers_survive(self):
        for head in (
            "### Frequently Asked Questions",
            "### Case Summary",
            "### Passage",
            "## Summary",
        ):
            t = head + "\n\nreal content"
            out, did = strip_wrap_preamble(t)
            self.assertFalse(did, f"{head!r} must NOT be stripped")
            self.assertEqual(out, t)

    def test_qa_format_never_matches(self):
        t = "Q: What is the capital?\nA: Paris.\n\nQ: And the currency?\nA: The euro."
        out, did = strip_wrap_preamble(t)
        self.assertFalse(did)
        self.assertEqual(out, t)

    def test_300_char_cap(self):
        t = "Here is the rewritten passage " + "y" * 300 + "\n\nbody"
        out, did = strip_wrap_preamble(t)
        self.assertFalse(did)

    def test_no_blank_line_means_no_strip(self):
        out, did = strip_wrap_preamble("Here is the rewritten passage: body with no break")
        self.assertFalse(did)


class TestRuleDispatch(unittest.TestCase):
    def test_non_wrap_arms(self):
        for s in ("quality-first", "diversity-oriented", "disagreement-aware", "rewire-inspired"):
            self.assertIs(rule_for(s, "p1"), strip_wiki, s)
            self.assertIs(rule_for(s, "distill"), strip_distill_preamble, s)

    def test_wrap_arm_uses_its_own_rule_for_both_passes(self):
        self.assertIs(rule_for("wrap-inspired", "p1"), strip_wrap_preamble)
        self.assertIs(rule_for("wrap-inspired", "distill"), strip_wrap_preamble)


if __name__ == "__main__":
    unittest.main()
