"""STEP 3 -- copy the shared base, mix in the strategy half, shuffle to the training input.

Ported from 10_postprocess/03_mix_shared_top*.py and 05_mix_shared_top_rewrite.py:

  * the shared base is written with `source_prompt = "original"` and its ORIGINAL text
  * `doc_id` overlap between the base and the strategy half's SOURCE documents must be 0
  * document-level shuffle, seed 42, via io.bucketed_shuffle (the shuffle-parity contract)

Output schema of `shuffled/`: the training input.  `orig_doc_id` is the source document,
`text` is what the model trains on, `source_prompt` says where it came from.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from ..io import bucketed_shuffle, check

FINAL_SCHEMA = pa.schema(
    [
        pa.field("orig_doc_id", pa.int64()),
        pa.field("text", pa.large_string()),
        pa.field("source_prompt", pa.large_string()),  # original | wikipedia | distill | wrap_<style>
        pa.field("tokens_llama2", pa.int32()),         # length of THIS row's text, no BOS
    ]
)


def base_table(doc_ids, texts, tokens) -> pa.Table:
    n = len(doc_ids)
    return pa.table(
        {
            "orig_doc_id": pa.array(np.asarray(doc_ids, dtype=np.int64), type=pa.int64()),
            "text": pa.array(list(texts), type=pa.large_string()),
            "source_prompt": pa.array(["original"] * n, type=pa.large_string()),
            "tokens_llama2": pa.array(np.asarray(tokens, dtype=np.int32), type=pa.int32()),
        },
        schema=FINAL_SCHEMA,
    )


def strategy_table(doc_ids, texts, tokens, source_prompt) -> pa.Table:
    n = len(doc_ids)
    sp = [source_prompt] * n if isinstance(source_prompt, str) else list(source_prompt)
    check(len(sp) == n, "source_prompt length mismatch")
    return pa.table(
        {
            "orig_doc_id": pa.array(np.asarray(doc_ids, dtype=np.int64), type=pa.int64()),
            "text": pa.array(list(texts), type=pa.large_string()),
            "source_prompt": pa.array(sp, type=pa.large_string()),
            "tokens_llama2": pa.array(np.asarray(tokens, dtype=np.int32), type=pa.int32()),
        },
        schema=FINAL_SCHEMA,
    )


def check_no_overlap(base_doc_ids: np.ndarray, strategy_source_doc_ids: np.ndarray) -> int:
    """The 1.5B Step-3 gate: 0 doc_id overlap between the base and the strategy sources."""
    inter = np.intersect1d(
        np.unique(np.asarray(base_doc_ids, np.int64)),
        np.unique(np.asarray(strategy_source_doc_ids, np.int64)),
        assume_unique=True,
    )
    check(inter.size == 0, f"{inter.size:,} doc_id overlap between the shared base and the strategy half")
    return 0


def shuffle_final(in_paths, out_dir, tmp_dir, seed=42, rows_per_out_shard=500_000, mem_gb=240.0):
    return bucketed_shuffle(
        in_paths, out_dir, tmp_dir, seed=seed,
        rows_per_out_shard=rows_per_out_shard, mem_gb=mem_gb,
    )


def horizon_plan(corpus_tokens: int, cfg) -> dict:
    """The training handoff (Decision 5 / plan.md 18.3).

    Horizons are MATCHED TOKEN horizons across all six settings.  A corpus slightly under
    `epoch_tokens` is simply re-read slightly more often; the horizon is NEVER computed as
    corpus_tokens * n_epochs.
    """
    t = cfg.budgets["training"]
    horizons = [int(h) for h in t["horizons"]]
    return dict(
        corpus_tokens=int(corpus_tokens),
        epoch_tokens=int(t["epoch_tokens"]),
        horizons=horizons,
        passes_to_horizon=[round(h / corpus_tokens, 6) for h in horizons],
        note=(
            "Cumulative TRAINING-TOKEN horizons, matched across all six settings. "
            "A corpus under epoch_tokens is re-read slightly more often; the horizon is "
            "never corpus_tokens * n_epochs."
        ),
        seeds=[int(s) for s in t["seeds"]],
    )
