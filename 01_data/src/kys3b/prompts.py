"""Prompt loading, the md5 provenance gate, and the templated-overhead gate.

The six prompt files are byte-copies of the 1.5B production set.  Their md5s and their
templated empty-document token counts are BOTH asserted, in that order, before any GPU
time is spent.  A prompt that differed by one byte or one token would not reproduce these.

WRAP style assignment is keyed only on (42, shard_index) -- worker-independent and
resume-safe.  The key order easy,hard,wiki,qa is part of the seed: reordering it silently
re-rolls which document gets which style.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .config import CODE_ROOT, WRAP_STYLES
from .io import check

PROMPT_DIR = CODE_ROOT / "prompts"
PLACEHOLDER = "[TEXT]"

# md5 of every production prompt, from projects/rewrite/prompts/README.md, re-verified
# against the copies in this repo on 2026-09-23.
MD5 = {
    "p1_wiki": "bca104fe6e298615e5ccb9c9c747073b",
    "p2_distill": "538700534e99d5e80b268fd9b2408b48",
    "wrap_easy": "0735f53aca80cadaa8d67727680dbbfd",
    "wrap_hard": "e99a613bcd4146416428d576af6f200a",
    "wrap_wiki": "cec46736de0229e6d7a0f022cd2e661a",
    "wrap_qa": "733fbeea43050cb4a4e27f9384b9014e",
}
FILES = {
    "p1_wiki": PROMPT_DIR / "wikipedia_style_rephrasing_grounded.md",
    "p2_distill": PROMPT_DIR / "distill" / "distill_prompt.txt",
    "wrap_easy": PROMPT_DIR / "wrap" / "easy.txt",
    "wrap_hard": PROMPT_DIR / "wrap" / "hard.txt",
    "wrap_wiki": PROMPT_DIR / "wrap" / "wiki.txt",
    "wrap_qa": PROMPT_DIR / "wrap" / "qa.txt",
}


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def load_text(key: str) -> str:
    p = FILES[key]
    check(p.exists(), f"missing prompt file {p}")
    return p.read_text()


def check_md5() -> dict:
    """Hard gate: every prompt is byte-identical to the 1.5B production set."""
    out = {}
    for key, want in MD5.items():
        got = _md5(load_text(key))
        check(got == want, f"prompt {key}: md5 {got} != expected {want} ({FILES[key]})")
        out[key] = got
    wp = json.loads((PROMPT_DIR / "wrap_prompts.json").read_text())
    check(
        list(wp) == WRAP_STYLES,
        f"wrap_prompts.json key order {list(wp)} != {WRAP_STYLES} -- the order is part of the seed",
    )
    for s in WRAP_STYLES:
        check(
            wp[s] == load_text(f"wrap_{s}"),
            f"wrap_prompts.json[{s}] differs from prompts/wrap/{s}.txt",
        )
    return out


def wrap_prompts() -> dict[str, str]:
    return json.loads((PROMPT_DIR / "wrap_prompts.json").read_text())


def build_content(mode: str, doc_text: str, template: str | None, wp: dict | None, style: str | None) -> str:
    """The user-message content for one document, pre chat-template.

    Verbatim from 07_rewrite/rewrite_worker.py:46-52:
      grounded -> template.replace("[TEXT]", doc_text)
      wrap     -> wrap_prompts[style] + doc_text   (the style string already ends "Passage:\\n")
    """
    doc_text = doc_text or ""
    if mode == "grounded":
        return template.replace(PLACEHOLDER, doc_text)
    check(mode == "wrap", f"unknown prompt mode {mode!r}")
    return wp[style] + doc_text


def assign_wrap_styles(shard_index: int, n_rows: int, base_seed: int = 42) -> list[str]:
    """Deterministic per-(shard,row) style assignment.

    Verbatim from 07_rewrite/rewrite_worker.py:54-62.  Seeded only by (base_seed,
    shard_index), so row i of a given shard ALWAYS maps to the same style regardless of
    which worker runs it or when; the whole shard is drawn in one call, so there is no
    partial-consumption state to lose mid-shard.
    """
    rng = np.random.default_rng([base_seed, shard_index])
    idx = rng.integers(0, len(WRAP_STYLES), size=n_rows)
    return [WRAP_STYLES[i] for i in idx]


def overhead_specs(mode: str) -> list[tuple[str, str]]:
    """(label, content-for-an-empty-document) pairs a job of this mode can emit.

    A grounded job has one; the wrap styled pass has FOUR, and all four are asserted --
    one value must not stand in for four.
    """
    if mode == "grounded_p1":
        return [("p1_wiki", build_content("grounded", "", load_text("p1_wiki"), None, None))]
    if mode == "distill":
        return [("p2_distill", build_content("grounded", "", load_text("p2_distill"), None, None))]
    if mode == "wrap":
        wp = wrap_prompts()
        return [(f"wrap_{s}", build_content("wrap", "", None, wp, s)) for s in WRAP_STYLES]
    raise KeyError(mode)


def check_overheads(tokenizer, cfg, modes=("grounded_p1", "distill", "wrap")) -> dict:
    """Measure and assert the templated empty-document length for every prompt text."""
    want = cfg.vllm["prompt_overheads"]
    chat = cfg.vllm["chat"]
    out = {}
    for mode in modes:
        for label, content in overhead_specs(mode):
            s = tokenizer.apply_chat_template(
                [{"role": chat["role"], "content": content}],
                add_generation_prompt=chat["add_generation_prompt"],
                tokenize=False,
            )
            got = len(tokenizer(s, add_special_tokens=False).input_ids)
            check(
                got == int(want[label]),
                f"prompt overhead {label}: measured {got} != expected {want[label]} -- "
                "the chat template or the prompt changed; refusing to spend GPU time",
            )
            out[label] = got
    return out
