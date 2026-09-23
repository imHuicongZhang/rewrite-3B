r"""STEP 1 -- strip model-artifact preambles from rewritten text, then RECOUNT.

Every rule here is ported verbatim from the 1.5B pipeline.  All are START-ANCHORED: the body
is never touched by `.replace()` or `.lstrip()`.  Which rule applies to which (setting, pass)
is exactly what the 1.5B scripts did:

  non-WRAP arms, pass 1   `WIKI_PREFIX` slice + `strip_instruction_leak`
                          (01_strip_prefix.py:111-113, 01_strip_prefix_diversity.py, _rewrite.py)
  non-WRAP arms, distill  `strip_distill_preamble`  -- the shared 120-char extended rule
                          (pp_io.strip_distill_preamble)
  WRAP arm, both passes   `strip_wrap_preamble`     -- the 300-char OPENERS/STRICT_META rule
                          (01_strip_prefix_wrap.py:104-133)

After stripping, `rewritten_tokens` is recounted with the llama-2 tokenizer
(add_special_tokens=False, no BOS).  Re-running finds nothing left to strip.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..io import atomic_write_table

# --------------------------------------------------------------------------- non-WRAP pass 1
WIKI_PREFIX = "Here is a paraphrased version:\n\n"  # exact, case-sensitive, start-anchored
INSTRUCTION_LEAK_ANCHOR = "Important: Do not add any information, claims, or details that are not"


def strip_instruction_leak(text: str) -> tuple[str, bool]:
    """Remove a leaked rephrasing-instruction block from the START of `text`.

    Verbatim from pp_io.strip_instruction_leak.  If the whole document is the leak (no blank
    line), returns ('', True).
    """
    if text and text.startswith(INSTRUCTION_LEAK_ANCHOR):
        cut = text.find("\n\n")
        return (text[cut + 2 :], True) if cut >= 0 else ("", True)
    return text, False


def strip_wiki(text: str) -> tuple[str, bool]:
    did = False
    if text and text.startswith(WIKI_PREFIX):
        text = text[len(WIKI_PREFIX) :]
        did = True
    text, leak = strip_instruction_leak(text)
    return text, bool(did or leak)


# --------------------------------------------------------------------------- non-WRAP distill
DISTILL_MAX_PREAMBLE_CHARS = 120
DISTILL_PREAMBLE_WORDS = ("paraphras", "condensed", "rewrit", "summary", "version")
DISTILL_OPENERS = (
    "here is", "here's", "here are", "certainly", "sure", "below is",
    "of course", "the following is",
)


def strip_distill_preamble(text: str) -> tuple[str, bool]:
    """Verbatim from pp_io.strip_distill_preamble: first-paragraph rule, then first-line rule."""
    if not text:
        return text, False
    # (a) first-paragraph meta preamble
    cut = text.find("\n\n")
    if 0 <= cut <= DISTILL_MAX_PREAMBLE_CHARS:
        head = text[:cut].strip()
        low = head.lower()
        has_word = any(w in low for w in DISTILL_PREAMBLE_WORDS)
        if (
            (head.startswith(("###", "##", "**")) and has_word)
            or (low.startswith(DISTILL_OPENERS) and has_word)
            or (low.startswith("paraphrased") and head.endswith(":"))
        ):
            return text[cut + 2 :], True
    # (b) first-line meta preamble ending in ':'
    nl = text.find("\n")
    if 0 <= nl <= DISTILL_MAX_PREAMBLE_CHARS:
        head = text[:nl].strip()
        low = head.lower()
        has_word = any(w in low for w in DISTILL_PREAMBLE_WORDS)
        if (
            head.endswith(":")
            and has_word
            and (
                head.startswith(("###", "##", "**"))
                or low.startswith(DISTILL_OPENERS)
                or low.startswith("paraphrased")
            )
        ):
            return text[nl + 1 :].lstrip("\n"), True
    return text, False


# --------------------------------------------------------------------------- WRAP, both passes
WRAP_MAX_PREAMBLE_CHARS = 300
WRAP_OPENERS = (
    "Here is", "Here's", "Here are", "Below is", "Below are",
    "Sure, here", "Sure! Here", "Sure, here's", "Sure thing, here",
    "Certainly! Here", "Certainly, here", "Of course! Here", "Of course, here",
    "I have rewritten", "I've rewritten", "I have reworded", "I've reworded",
    "The following is", "This is the rewritten", "This is a rewritten",
    "Rewritten passage", "Rewritten version", "Rewritten text",
    "Here is a paraphrased", "Here's the rewritten", "Here is the rewritten",
)
WRAP_SIGNAL_WORDS = (
    "passage", "rewritten", "rewrite", "reworded", "rephrased", "paraphrase",
    "version", "simplified", "simpler", "plain language", "neutral", "factual",
    "summary", "question", "q&a", "style", "reading level", "young child", "requested",
)
# Deliberately STRONGER than SIGNAL_WORDS so genuine content headers
# ("### Frequently Asked Questions", "### Case Summary", "### Passage") survive.
WRAP_STRICT_META = (
    "paraphras", "rewritten", "rewrite", "reworded", "rephrased",
    "simple version", "simpler version", "simplified version", "plain language",
    "condensed version", "scholarly language", "young child version",
)


def strip_wrap_preamble(text: str) -> tuple[str, bool]:
    """Verbatim from 01_strip_prefix_wrap.py:104-133.

    The qa 'Q:' opening never matches either branch, and content headers without a
    STRICT_META phrase are left untouched.
    """
    if not text:
        return text, False
    nl = text.find("\n\n")
    if nl < 0 or nl > WRAP_MAX_PREAMBLE_CHARS:
        return text, False
    head = text[:nl]
    low = head.lower()
    # (a) sentence-opener preamble (case-sensitive opener + any signal word)
    if head.startswith(WRAP_OPENERS) and any(k in low for k in WRAP_SIGNAL_WORDS):
        return text[nl + 2 :], True
    # (b) markdown/bold header or a bare 'Paraphrased ...'/'Rewritten passage ...' label,
    #     gated on a STRICT_META phrase
    if (
        head.lstrip().startswith(("###", "##", "**"))
        or low.startswith("paraphrased")
        or low.startswith("rewritten passage")
    ) and any(s in low for s in WRAP_STRICT_META):
        return text[nl + 2 :], True
    return text, False  # opens with content -> leave untouched


# --------------------------------------------------------------------------- dispatch
def rule_for(setting: str, pass_name: str):
    """Which strip rule the 1.5B pipeline applied to this (setting, pass)."""
    if setting == "wrap-inspired":
        return strip_wrap_preamble
    return strip_wiki if pass_name == "p1" else strip_distill_preamble


def strip_shard(path, setting: str, pass_name: str, ltok) -> dict:
    """Strip + recount one shard in place, atomically.  Idempotent."""
    rule = rule_for(setting, pass_name)
    t = pq.read_table(path, use_threads=False)
    status = t.column("status").to_numpy(zero_copy_only=False)
    texts = t.column("rewritten").to_pylist()
    toks = t.column("rewritten_tokens").to_numpy(zero_copy_only=False).astype(np.int64).copy()

    s2 = np.flatnonzero(status == 2)
    changed, tok_before, tok_after = [], 0, 0
    new_texts = list(texts)
    for j in s2.tolist():
        tok_before += int(toks[j])
        s, did = rule(texts[j])
        if did:
            new_texts[j] = s
            changed.append(j)
    if changed:
        from ..tokens import count_batch

        recounted = count_batch(ltok, [new_texts[j] for j in changed])
        for j, nt in zip(changed, recounted):
            toks[j] = nt
    tok_after = int(toks[s2].sum()) if s2.size else 0

    if changed:
        cols = {name: t.column(name) for name in t.schema.names}
        cols["rewritten"] = pa.array(new_texts, type=pa.large_string())
        cols["rewritten_tokens"] = pa.array(toks.astype(np.int32), type=pa.int32())
        atomic_write_table(pa.table(cols), path)

    return dict(
        n_status2=int(s2.size),
        n_stripped=len(changed),
        tok_before_s2=int(tok_before),
        tok_after_s2=int(tok_after),
        tok_saved=int(tok_before - tok_after),
    )
