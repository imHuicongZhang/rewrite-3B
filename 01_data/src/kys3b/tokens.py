"""Llama-2 token counting for REWRITTEN text.

The pool ships `tokens-llama2` for source text, so this module is only ever applied to
model output.  Recipe, verbatim from 03_TokenCounts/count_tokens.py:5:

    tokens = len(tokenizer(text, add_special_tokens=False))     # pure text, no BOS/EOS
    empty / None -> 0

TRAIN length adds one BOS: `tokens + 1`.  A slow tokenizer is refused.
"""
from __future__ import annotations

from functools import lru_cache

from .io import check

BOS = 1  # the +1 in every budget


@lru_cache(maxsize=2)
def load_llama2(path: str, expected_vocab_size: int = 32000):
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(path, use_fast=True)
    check(
        bool(getattr(tk, "is_fast", False)) and hasattr(tk, "backend_tokenizer"),
        f"llama-2 tokenizer at {path} is not FAST; refusing a slow tokenizer",
    )
    if expected_vocab_size:
        check(
            tk.vocab_size == expected_vocab_size,
            f"llama-2 vocab_size {tk.vocab_size} != expected {expected_vocab_size}",
        )
    return tk


def count_batch(tokenizer, texts: list[str]) -> list[int]:
    """Per-document llama-2 token counts (no specials).  None/'' -> 0."""
    idx = [i for i, t in enumerate(texts) if t]
    out = [0] * len(texts)
    if not idx:
        return out
    enc = tokenizer([texts[i] for i in idx], add_special_tokens=False)["input_ids"]
    for i, ids in zip(idx, enc):
        out[i] = len(ids)
    return out


def train_len(tokens) -> int:
    return int(tokens) + BOS
