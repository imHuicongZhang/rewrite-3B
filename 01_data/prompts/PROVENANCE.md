# Prompt provenance

These six files are **byte-copies** of the 1.5B production prompt set at
`projects/rewrite/prompts/`.  They are the prompts that produced the published 1.5B corpus.

| file | md5 | templated overhead | used by |
|---|---|---:|---|
| `wikipedia_style_rephrasing_grounded.md` | `bca104fe6e298615e5ccb9c9c747073b` | 150 | pass 1, the four non-WRAP arms |
| `distill/distill_prompt.txt` | `538700534e99d5e80b268fd9b2408b48` | 185 | pass 2, **all five** rewriting arms |
| `wrap/easy.txt` | `0735f53aca80cadaa8d67727680dbbfd` | 72 | WRAP pass 1, style `easy` |
| `wrap/hard.txt` | `e99a613bcd4146416428d576af6f200a` | 66 | WRAP pass 1, style `hard` |
| `wrap/wiki.txt` | `cec46736de0229e6d7a0f022cd2e661a` | 73 | WRAP pass 1, style `wiki` |
| `wrap/qa.txt` | `733fbeea43050cb4a4e27f9384b9014e` | 83 | WRAP pass 1, style `qa` |

`wrap_prompts.json` is assembled from the four `wrap/*.txt` files and is asserted to match them.

## Two gates, both enforced before any GPU time

1. **md5** (`kys3b.prompts.check_md5`) -- every file byte-identical to the table above, and
   `wrap_prompts.json` key order exactly `easy, hard, wiki, qa`.
2. **templated overhead** (`kys3b.prompts.check_overheads`) -- the templated length of an EMPTY
   document under the production Qwen2.5-7B-Instruct chat template must equal the
   `prompt_overheads` block of `configs/vllm.yaml`.  A prompt that differed by one byte, or a
   chat template that changed, would not reproduce these integers.

Both were verified on 2026-09-23 against the local
`models/Qwen2.5-7B-Instruct`: all six match.

## Key order is part of the seed

WRAP assigns one style per document with
`np.random.default_rng([42, shard_index]).integers(0, 4, n_rows)` indexing
`["easy", "hard", "wiki", "qa"]`.  **Reordering the keys silently re-rolls which document gets
which style.** `tests/test_wrap_styles.py` pins the resulting stream with golden vectors so a
NumPy PCG64 change fails the suite instead of quietly changing the corpus.

## The abandoned paper-verbatim WRAP set must never be used

A WRAP set copied verbatim from arXiv:2401.16380 App. G was tried first and **abandoned** after
a 100-document pilot (`easy` produced near-empty output on 32/100, `hard` failed on 22/100).  It
is distinguishable at a glance: its keys are `easy / **medium** / hard / qa`.  It survives at
`projects/rewrite/06_vllm/wrap_styles_sample.py:34-49`.  `medium` appears nowhere in the
production pipeline, and `tests/test_prompts.py` asserts it is absent here.

## Recovery history (1.5B)

Four of the six (`wrap/*`) plus `distill/distill_prompt.txt` were **recovered**, not copied: the
originals lived under `data_rewrite/prompts/` and that root was deleted.  They were reconstructed
from artefacts that embed them verbatim (fully chat-templated model inputs) and validated against
the empty-document overhead figures printed by the production workers at run time.  The full
account is in `projects/rewrite/prompts/README.md`.  A later independent byte-comparison against
the published originals (`rewrite-vllm` round 9) confirmed all six identical.
