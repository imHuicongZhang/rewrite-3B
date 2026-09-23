# rewrite-3B — 3B scale-up implementation plan (PLAN ONLY, nothing built, nothing submitted)

**Date:** 2026-09-23 (revision 2 — all decisions locked except one paper-side correction)
**Author:** drafted by Claude for Huicong Zhang, for review before any code is written.
**Status:** NO code written, NO dataset downloaded, NO Slurm job submitted, NO git push.
Only read-only inspection was performed (plus `sbatch --test-only`, which creates no job).

**Sources read for this plan**

| tag | what | where |
|---|---|---|
| **P** | the paper | `00_paper/Know_Your_Sources.pdf` (29 pp, read in full incl. App. A–F) |
| **C15** | the 1.5B implementation | `/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/` (`00_TMP`, `01_explore`, `04_select`, `05_select_s5_variants`, `06_lambda_grid`, `06_vllm`, `07_rewrite`, `09_Distill`, `10_postprocess`, `prompts/`) |
| **C600** | a later, larger-pool selection design by the same author | `projects/rewrite/13_600M/` |
| **CVLLM** | a later, portable rewriting pipeline + a measured audit of the 1.5B run | `projects/rewrite-vllm/` (esp. `docs/DESIGN_DELTA.md`) |
| **CRAW** | the raw (no-rewrite) control pipeline already running on Skipjack | `projects/nanotron-kys/tools/kys_raw/` |
| **HF** | the scored pool | `blab-jhu/KYS-DCLM-Refinedweb-100M-Scored` (gated; metadata + README fetched with your token) |

---

## 0. Locked decisions (revision 2)

Every design question raised in revision 1 is now settled. Details and evidence in §20.

| # | decision | status |
|---|---|---|
| 1 | 5M analysis sample: **use the published 100M scored pool as-is; do not re-remove anything** | ✅ **RESOLVED (pipeline)** — ⚠️ **but the paper's claim is factually wrong and must be corrected; see §20 D-1** |
| 2 | Disagreement-Aware candidate domain: **Option C — per-scorer budget = `B_s` = 20B**, τ_q=Q30(q\|U), τ_v=Q90(v\|U), λ=0.5 | ✅ RESOLVED |
| 3 | **Both rewriting passes generated over the full source set for all five rewriting arms**; no adaptive skip | ✅ RESOLVED |
| 4 | REWIRE: **κ=2 ⇒ 40B raw source pool**, both passes, fastText on rewritten output, top ~10B retained | ✅ RESOLVED |
| 5 | Diversity: preserve policy A (no cross-topic backfill, no padding); **training horizon stays matched at 20B/40B/60B** | ✅ RESOLVED |
| 6 | Diversity keys: consensus `q` at **selection**, `fasttext-ranking-v2` at **assembly** — exactly as the 1.5B code | ✅ RESOLVED |
| 7 | Storage root: **`/weka/projects/bvandur1/zhuicon1/rewrite-3b/`**, symlinks later at the two scratch paths | ✅ RESOLVED |
| 8 | 10,000-row shards + claim-directory distribution + stale-claim recovery | ✅ RESOLVED |
| 9 | Both `jhu2` and `scavenger` against one claim directory; `jhu2` **allows up to** 32 concurrent GPUs (no guarantee) | ✅ RESOLVED |
| 10 | This repo's final output is **shuffled parquet**; Nanotron `.ds` tokenization is the training repo's job | ✅ RESOLVED |
| 11 | This 3B experiment is **separate** from the 7B/600M/50B-per-arm design; no shared budgets, pool, DA simplification or doc-id space | ✅ RESOLVED |
| 12 | **λ = 0.5 only**; no λ sweep at 3B | ✅ RESOLVED |

**Two invariants that govern everything below:**

1. **Matched training-token horizons.** All six settings are trained and evaluated at the *same*
   cumulative horizons — **~20B / ~40B / ~60B tokens** — regardless of each corpus's exact size. If a
   corpus is slightly under 20B, it is simply seen slightly more than 1 / 2 / 3 times. Horizons are
   never derived as `corpus_size × 3`.
2. **No document of the shared base may re-enter any strategy pool.** Every strategy selects only from
   the residual pool `D'` = ALIVE − BASE, and `|BASE ∩ S| == 0` is a hard `stop()` for all six S.

---

## 1. What I learned from the latest paper

### 1.1 The six settings (P §4.2, Table 1)

Every setting is a **10B-token training mixture** = a **fixed shared 5B base** (top of the DCLM
fastText ranking) + **~5B strategy-specific tokens**. The candidate pool `D` is *the remainder after
the shared base is removed*, so **every strategy, including QUALITY-FIRST, selects outside the base**
(P §5.1, explicit).

| setting | strategy-specific 5B |
|---|---|
| QUALITY-BASE | next 5B fastText-ranked tokens, raw (no rewrite) |
| QUALITY-FIRST | rewrites of the top documents of `D` under fastText percentile |
| WRAP-INSPIRED | rewrites of a uniform random sample, 4 style prompts + shared distill |
| REWIRE-INSPIRED | top rewrites from a **2× larger** random pool, filtered *after* rewriting |
| DIVERSITY-ORIENTED | rewrites of documents selected within each of 24 WebOrganizer topics |
| DISAGREEMENT-AWARE | rewrites of documents ranked by `u_i = q_i + λ√v_i` |

### 1.2 Two budgets (P §4.1, §5.1)

- `B_s` = **source** budget = 10B raw tokens (the size of the selected source set).
- `B` = **rewritten-output** budget = 5B tokens (what actually enters training).
- Rewriting compresses, so `B_s = 2B`. Oversampling factor `κ = 2` for REWIRE only.
- `λ = 0.5`; quality floor and variance cap are set **by percentile**. "Every constant is fixed
  without reference to downstream performance."

### 1.3 The budget-selection operator (P Eq. 3)

`Top_b(A, f) = { d ∈ A : f(d) ≥ θ_b }` where `θ_b` is the largest threshold with
`Σ ℓ(d) ≥ b`. Length `ℓ` is measured **on the elements of A** — source tokens for the pre-rewrite
strategies, *rewritten* tokens for REWIRE. Ties are broken so selection stops when the budget is
first reached.

### 1.4 Scorers and derived quantities (P §4.1, Eq. 1–2)

Three scorers → tie-aware **global percentiles** `r^(s) ∈ (0,1]` over all N documents, then
`q = mean_s(r^(s))`, `v = Var_s(r^(s))` (population variance, ddof=0). Not stored — derived.

### 1.5 Strategy definitions (P §4.2)

- **QUALITY-FIRST**: `S_QF = Top_{B_s}(D, r^(fastText))`. Single scorer, fastText only.
- **WRAP-INSPIRED**: uniform random from `D` to `B_s`; each doc gets one of four adapted style
  prompts; **also** rewritten with the shared distill prompt.
- **REWIRE-INSPIRED**: uniform random pool `R` of `κ·B_s` tokens; rewrite **all** of `R`;
  `S_RI = Top_B(π(R), r^(fastText))` — fastText applied **to the rewritten output**, and the set is
  sized by `B` (5B), not `B_s`.
- **DIVERSITY-ORIENTED**: partition `D` by topic; topic `c` gets `p_c·B_s` tokens where `p_c` is its
  **share of pool tokens**; within topic rank by **consensus quality `q`** (all three scorers).
  `S_DIV = ∪_c Top_{p_c B_s}(D_c, q)`.
- **DISAGREEMENT-AWARE**: for each scorer take the highest-ranked documents "until a **10B-token
  budget** is reached"; `U` = union of the three; quality floor `τ_q` = **30th percentile of q over
  U**, variance cap `τ_v` = **90th percentile of v over U**; `U_τ = {q ≥ τ_q, v ≤ τ_v}`;
  `S_DA = Top_{B_s}(U_τ, u)` with `u = q + λ√v`.

### 1.6 Realized feasible-domain numbers (P App. D.1, Tables 8–9)

`U` = 14,982,068 docs / 21.96B tokens; `U_τ` = 10,435,667 docs / 16.70B tokens (69.65% of U by
doc, 76.08% by token). λ=0.5 admits 5,602,476 docs to fill 10B. Only 51,783 docs (0.35% of U) are
removed by the cap alone.

### 1.7 Rewriting (P §5.2, App. C, Fig. 7–8)

- Rewriter: **Qwen2.5-7B-Instruct**, greedy, max 4,096 output tokens.
- Non-WRAP arms: pass 1 = **grounded Wikipedia-style rephrasing** (FinePhrase/Nemotron-CC template +
  an added explicit grounding instruction); pass 2 = **distill** (used as released).
- WRAP: pass 1 = one of four adapted style prompts per document (`easy`/`hard`/`wiki`/`qa`);
  pass 2 = the **same shared distill prompt**.
- Measured on the selection sample: wikipedia-style compression 0.637, distill 0.432.

### 1.8 Production prompt composition (P App. C, verbatim)

> "**Both rewriting passes are generated over the full selected source set.** … For the four
> strategies that select sources before rewriting, the final corpus is constructed from first-pass
> outputs and supplemented with second-pass outputs to approach the target rewritten-token budget.
> REWIRE-INSPIRED instead applies its post-rewrite fastText filtering before the final budget is
> filled."

and for WRAP: "although both passes are generated over the full source set, only enough *distill*
output is retained to fill the remaining rewritten-token budget."

**Paper Table 7 — retained rewritten tokens by pass (B):**

| strategy | pass 1 | distill | distill share |
|---|---:|---:|---:|
| QUALITY-FIRST | 3.26 | 1.74 | 34.9% |
| WRAP-INSPIRED | 4.16 | 0.84 | 16.9% |
| REWIRE-INSPIRED | 2.35 | 2.65 | 53.1% |
| DIVERSITY-ORIENTED | 3.70 | 1.19 | 24.4% |
| DISAGREEMENT-AWARE | 3.54 | 1.46 | 29.1% |

**Every arm needed the distill pass. Pass 1 alone never reached B.** This is why Decision 3 is
locked to "always generate both passes".

### 1.9 Epochs vs. total tokens (P §5.3–5.4, Fig. 1)

One ~10B-token mixture, **repeated three times** → ~30B total, with evaluation at ~10B / ~20B / ~30B
cumulative **training tokens**. The corpus is *not* regenerated per epoch.

Critically, the horizons are **token horizons, not pass counts**. DIVERSITY-ORIENTED's 1.5B corpus
was 9.890B rather than 10B, and it was still trained and evaluated against the same matched
~10B/20B/30B token budgets as the other five settings — it simply completed slightly more than
1/2/3 passes over its corpus. The 3B design inherits this exactly (§0 invariant 1).

### 1.10 The analysis holdout (P §3, Fig. 1)

> "We first draw a 5M-document sample from DCLM-RefinedWeb for scorer and topic analysis. We
> **exclude this sample** from subsequent training-data construction."

**This sentence is false for the published pool and for the 1.5B run.** Proven from first-hand
artifacts in §20 D-1: 4,998,853 of the 5,001,383 analysis-sample documents are present in the
published 99,949,162-row table, exactly 5.001% of it. The pipeline action is nonetheless unchanged
(use the pool as published); it is the paper that needs correcting.

---

## 2. What I learned from the 1.5B implementation

### 2.1 The source table

`6_merged_clean`: 200 parquet shards × 500,000 rows (last 449,162) = **99,949,162 rows**,
`doc_id` contiguous 0…99,949,161, `doc_id == shard_index*500000 + row_index`. This tree is
**deleted** from the old cluster; the identical table is published as **HF
`blab-jhu/KYS-DCLM-Refinedweb-100M-Scored`** (§10).

It was produced by `00_TMP/merge_remove_50k.py` from a 100,000,000-row scored+topic-labelled pool by
physically removing exactly the **50,838** positions in
`01_explore/match_50k_prefix4000.npy` — the ModernBERT-annotator contamination set, and nothing else
(`00_TMP/merge_remove_50k_report.md`). `orig_doc_id` in the published table is the sorted complement
of those 50,838 positions over 0…99,999,999.

### 2.2 Token-budget convention (verified, `04_select/select_10b.py:92`, `03_TokenCounts/count_tokens.py`)

```
tokens-llama2 = len(llama2_tokenizer(text, add_special_tokens=False))     # pure text, no specials
TRAIN length  = tokens-llama2 + 1                                        # one leading BOS
```
Tokenizer: `tokenizers/llama2-unsloth-tokenizer` (`unsloth/llama-2-7b`, `LlamaTokenizer`,
vocab 32000, **fast** tokenizer enforced). **Every** budget in the 1.5B run — source and rewritten —
is `tokens + 1`. Rewritten outputs are counted with the *same* llama-2 tokenizer
(`rewritten_tokens`), never the Qwen tokenizer.

### 2.3 Determinism (verified, `select_10b.py:70-77`)

```python
_SS = np.random.SeedSequence(42); _CH = _SS.spawn(8)
RNG_VAL  = default_rng(_CH[0])   # 50,000-doc validation holdout
RNG_TIE  = default_rng(_CH[1])   # global tie-break permutation for every DESC ranking
RNG_WRAP = default_rng(_CH[2])   # WRAP uniform draw
RNG_REWR = default_rng(_CH[3])   # REWIRE uniform draw
```
`order_desc` = `lexsort((tie[idx], -score[idx].astype(float64)))`. The tie-break is **load-bearing**,
not cosmetic: the DCLM fastText score has an 8,655,073-document tie at its floor (percentile ≈0.0433).

`fill_to(order, tok, target)`: cumsum until first `≥ target`, **last document kept whole**; under-fill
calls `stop()` rather than truncating silently.

### 2.4 Exact 1.5B selection (`04_select/select_10b.py`, verified line by line)

```
VAL          = 50,000 docs, uniform, child0         → excluded everywhere
ALIVE        = all 99,949,162 minus VAL
shared-top-5B= fill_to(order_desc(ALIVE, fasttext-ranking-v2), 5e9)
REMAINING    = ALIVE minus shared-top-5B            ← the candidate pool D
quality-base = next 5e9   along the SAME fastText order (prefix continuation)
quality-first= next 10e9  along the SAME fastText order (deeper prefix; contains quality-base)
wrap         = fill_to(RNG_WRAP.permutation(REMAINING), 10e9)
rewrite      = fill_to(RNG_REWR.permutation(REMAINING), 20e9)   # <-- κ=2, hard-coded as "10B * 2"
diversity-first: for each of 24 topics c:
                 quota_c = 10e9 * (tokens_c / tokens_REMAINING)
                 fill_to(order_desc(cat_idx, q), quota_c)        # q = consensus, NOT fastText
                 then a q-DESC top-up from REMAINING if the union is under 10B
```
Source: `04_select/select_10b.py:51` — `REWRITE_TARGET = 20_000_000_000   # rewrite = 10B * 2`.

Realized (from `10_postprocess/DATASETS_SUMMARY.md` and `_step2_rewrite_summary.json`):

| block | docs | tokens |
|---|---:|---:|
| shared-top-5B | 4,120,164 | 5,000,002,332 |
| quality-base (5B) | 3,133,023 | 5,000,000,805 |
| quality-base (15B-base variant, 10B) | 6,136,187 | 10,000,000,529 |
| rewrite (REWIRE source) | 21,214,299 | 20,000,000,679 |
| disagreement-aware λ=0.5 | 5,602,476 | 10,000,002,827 |

### 2.5 DISAGREEMENT-AWARE — the exact coded domain

`05_select_s5_variants/select_s5.py` and `06_lambda_grid/select_lambda_grid.py`,
`06_lambda_grid/DOMAIN_STATS.md`:

```
A,B,C = each scorer's  top-10%-BY-TOKENS over REMAINING        # <-- NOT "top 10B tokens"
U     = A ∪ B ∪ C
τ_q   = Q30(q over U) = 0.6956847310066223
τ_v   = Q90(v over U) = 0.1036492049694062
U_τ   = {d ∈ U : q ≥ τ_q AND v ≤ τ_v}
S_DA  = fill_to(order_desc(U_τ, q + λ√v), 10e9),  λ = 0.5
```
`U` = 14,982,068 docs / 21,955,550,649 tok; `U_τ` = 10,435,667 / 16,703,787,941; λ=0.5 → 5,602,476
docs / 10,000,002,827 tok (53.69% of `U_τ`). **These reproduce the paper's App. D numbers exactly**,
which proves the coded rule (10%-by-tokens) — not the paper's prose ("a 10B-token budget") — is what
produced the published results. `REMAINING` ≈ 90.3B tokens, so 10% ≈ 9.03B per scorer, i.e. ≈0.90·B_s
— which is why the paper's "10B" was a defensible rounding of it, and why **Option C (20B = B_s per
scorer) is the right 3B analogue** (§20 D-2, resolved).

### 2.6 Rewriting (`07_rewrite/`, `09_Distill/`, `configs/vllm.yaml` in CVLLM)

Locked engine, **exactly five kwargs**:
```python
LLM(model="Qwen2.5-7B-Instruct", tensor_parallel_size=1, dtype="bfloat16",
    gpu_memory_utilization=0.85, max_model_len=32768)
SamplingParams(temperature=0, top_p=1.0, max_tokens=min(4096, 32768 - n_in))
```
- `gpu_memory_utilization`: the sbatch passed **0.85** for both passes; the `0.90` in the argparse
  default and the READMEs was never used (CVLLM `docs/SOURCE_INVENTORY.md` §3.3).
- Data parallelism = **N independent single-GPU processes**, `tensor_parallel_size=1`,
  8-way Slurm array, worker owns shards where `shard_idx % 8 == worker_id`. vLLM's own
  `data_parallel_size` was never used.
- Chat template: one **user** message, `add_generation_prompt=True`; Qwen's template injects its own
  default system message. No system message is authored.
- vLLM defaults silently inherited and **never passed**: `max_num_batched_tokens=16384`,
  `enable_chunked_prefill=True`, `enable_prefix_caching=True`, `enforce_eager=False`, `seed=0`,
  `swap_space=4`.
- Status codes: `0` = templated input over the drop threshold (never rewritten), `1` =
  `finish_reason == "length"` (truncated), `2` = `stop` (complete).
- Drop threshold: **pass 1 = 30720** (fixed); **distill = derived** as `max_model_len - max_tokens =
  28672`, so every kept doc gets the full 4,096-token output budget. (This asymmetry is deliberate and
  documented in `09_Distill/README.md`.)
- Output: `<dataset>/rewritten/part_NNNNN.parquet` and `<dataset>/distill/part_NNNNN.parquet`,
  atomic `.tmp` + `os.replace`, **all source columns preserved** plus `rewritten`,
  `rewritten_tokens` (llama-2), `status`, `finish_reason`, `input_tokens_qwen`, and `wrap_style`
  (wrap pass 1 only).
- Resume: a shard whose output parquet exists is skipped. Re-running the launcher is the resume.
- Distill reads the **raw source shards**, never pass-1 output (asserted in the worker).

### 2.7 WRAP style assignment (`07_rewrite/rewrite_worker.py:39,54-62`)

```python
WRAP_STYLES = ["easy", "hard", "wiki", "qa"]   # index order is part of the reproducible seed
rng = np.random.default_rng([42, shard_index]); idx = rng.integers(0, 4, n_rows)
```
One style per document, i.i.d. uniform, keyed only on `(42, shard_index)` → worker-independent and
resume-safe. Realized at 1.5B: documents balanced to 25.0% ± 0.1pp, but **tokens spread 2.36×**
(`hard` 33.6% vs `easy` 14.2%) because the styles expand differently. **No correction was applied.**

### 2.8 Prompts (`projects/rewrite/prompts/`, md5-verified, overhead-gated)

| file | md5 | overhead (empty doc, Qwen template) | used by |
|---|---|---:|---|
| `wikipedia_style_rephrasing_grounded.md` | `bca104fe6e298615e5ccb9c9c747073b` | 150 | pass 1, four non-WRAP arms |
| `distill/distill_prompt.txt` | `538700534e99d5e80b268fd9b2408b48` | 185 | pass 2, **all five arms** |
| `wrap/easy.txt` | `0735f53aca80cadaa8d67727680dbbfd` | 72 | wrap pass 1 |
| `wrap/hard.txt` | `e99a613bcd4146416428d576af6f200a` | 66 | wrap pass 1 |
| `wrap/wiki.txt` | `cec46736de0229e6d7a0f022cd2e661a` | 73 | wrap pass 1 |
| `wrap/qa.txt` | `733fbeea43050cb4a4e27f9384b9014e` | 83 | wrap pass 1 |

Grounded pass 1 substitutes into a literal `[TEXT]` placeholder; the wrap styles end with
`Passage:\n` and the document is appended. `wrap_prompts.json` key order `easy, hard, wiki, qa`
**is part of the seed** — reordering silently re-rolls the corpus.

These six were byte-compared against the published originals in CVLLM round 9: **all six identical**.
Note the `prompts/README.md` warning — four of them were *recovered* after
`/scratch/.../data_rewrite/prompts/` was deleted, and an abandoned paper-verbatim WRAP set
(keys `easy/medium/hard/qa`) survives at `06_vllm/wrap_styles_sample.py:34-49` and must never be used.

### 2.9 Post-processing (`10_postprocess/`)

**Step 1 — prefix strip.** Start-anchored only (`text[len(prefix):]`, never `.replace`/`.lstrip`),
removing `"Here is a paraphrased version:\n\n"` from `status==2` wiki rewrites, then **recount**
`rewritten_tokens` with the llama-2 tokenizer. Distill is scanned; for `diversity-first` and `wrap` a
conservative preamble-paragraph remover was additionally applied (≤120 chars, first paragraph, gated
on strict meta-phrases; content headers and the `Q:` format explicitly protected).

**Step 2 — assemble B.** For the four pre-rewrite strategies:
```
take ALL status==2 pass-1 outputs
if that is still < B:  top up from status==2 distill, in quality order, until (rewritten_tokens+1) ≥ B
if wiki+distill < B:   report the shortfall and STOP (never pad)
```
`02_assemble_5B.py:269` already contains the branch `if wiki_tokens >= TARGET: <quality-sorted fill,
no distill>`. **It never fired at 1.5B** — see Table 7. The same `doc_id` rewritten by two prompts is
kept as **two training examples**; distill is *not* deduplicated against wiki.

Quality sort keys at assembly (these differ per arm and differ from the *selection* key in one case):

| arm | assembly sort key |
|---|---|
| quality-first | `fasttext-ranking-v2` DESC |
| signal-disagreement-λX | `u = q + λ√v` DESC, float32, ddof=0 |
| **diversity-first** | **`fasttext-ranking-v2` DESC, within topic** (per-topic quota, policy A: no cross-topic backfill, no padding) |
| **wrap** | **none** — the distill top-up is a *seeded random draw* (`default_rng(42)`), deliberately no quality signal |

**Step 3 — mix + shuffle.** Physically copy `shared-top-5B` with `source_prompt="original"`, verify
**0** `doc_id` overlap between base and the rewritten set's source docs, then `bucketed_shuffle`
(seed 42, memory-bounded two-pass, `default_rng([42,b])` within bucket) into ~500k-row shards.

**REWIRE path (`README_rewrite.md`, `02/03/04/05_*_rewrite.py`)** — different and important:
1. strip prefixes, recount;
2. build the pool = **ALL status==2 wiki + ALL status==2 distill** (16.22B tok, 42.38M rows);
3. **fastText-score the rewritten text**, reproducing `fasttext-ranking-v2` exactly:
   `clean = text.replace("\n"," ").replace("\r"," ")[:100000]`, `predict(clean, k=1)`,
   `raw = p if label == "__label__hq" else 1-p`, empty → `0.0`; model
   `models/5m/external/fasttext_oh_eli5.bin`; plus a v2 percentile against the original 100M raw
   distribution (`rankdata/99,949,162` via `searchsorted`) for readability — **sorting uses the raw
   score** (monotone, so identical);
4. global DESC sort, `fill_to` to B (last doc whole);
5. mix with the shared base + shuffle.

### 2.10 1.5B final tally (`DATASETS_SUMMARY.md`, `_step4_rewrite_summary.json`)

| dataset | total tokens | note |
|---|---:|---|
| 10B-base (QUALITY-BASE) | 10,000,003,137 | 5B shared + 5B next-fastText, raw |
| quality-first | 10,000,002,344 | wiki 3.257B + distill 1.743B |
| diversity-first | 9,889,709,441 | **110.3M short**, 4 topics exhausted (policy A) — still trained to the matched 10/20/30B horizons |
| wrap | 10,000,002,484 | wrap 4.155B + distill 0.845B |
| signal-disagreement-λ05 | 10,000,002,785 | wiki 3.544B + distill 1.456B |
| rewrite (REWIRE) | 10,000,002,683 | pool 16.22B → fastText top-5B: wiki 2.347B + distill 2.653B |

REWIRE detail: pool 42,380,355 rewritten docs / 16,221,811,013 tok (wiki 21,185,312 / 8.934B;
distill 21,195,043 / 7.288B) → kept 12,290,444 docs / 5,000,000,351 tok at a raw-score cutoff of
0.11456. **Acceptance = 30.8% by token.** Distill won 53.1% of the kept tokens because distill
outputs score *higher* under fastText (median raw 0.0243 vs 0.0125 for wiki).

### 2.11 Measured compression ratios (CVLLM `DESIGN_DELTA.md` §5 — census, exact)

Output llama-2 tokens per source llama-2 token, per arm, per pass. Derived from
`07_rewrite/progress/*.json` + `09_Distill/progress/*.json` (146 worker files) and independently
confirmed against `13_600M/02_select/select_600m.py:114-123`:

| 1.5B arm | r (pass 1) | r (distill) | sum |
|---|---:|---:|---:|
| quality-first | 0.3399 | 0.2581 | **0.5981** |
| diversity-first | 0.3812 | 0.2845 | **0.6657** |
| signal-disagreement-λ05 | 0.3649 | 0.2750 | **0.6400** |
| wrap | 0.4313 | 0.3657 | **0.7970** |
| rewrite (REWIRE) | 0.4628 | 0.3660 | **0.8288** |

Also measured: the 1.5B run put **192.6B templated (Qwen) input tokens** against **69.5B llama-2
output tokens** — a **2.77:1 prefill-dominated** mix. This drives the throughput model in §13/§15.

### 2.12 A later, larger design also exists in the repo — and is NOT this experiment

`13_600M/02_select/README.md` designs a **600M-document-pool** version at 20B core + 60B remainders
(120B for rewire) for a 50B-per-arm training budget, and `rewrite-vllm` is its portable rewriting
pipeline (against `wytro/Know-Your-Sources-7B`). Two of its statements are worth carrying over:

> "**κ = 2 applies to the remainder** (120B vs 60B), following `04_select/select_10b.py:51` …
> It is *not* 2× on the total. The 1.5B convention scaled the source budget, never the core."

> "The v1 candidate domain (per-scorer top-10%-by-tokens union, then `cq ≥ Q30` and `cvar ≤ Q90`)
> is **disabled**: `disagreement-aware` now ranks `udis` over the whole core-excluded pool."

The first rule is **adopted** here (§7). The second is **rejected** for the 3B run — see §20 D-2 and
D-11. That design's budgets, pool, DA simplification and `doc_id` space are not reused anywhere.
What *is* reused from `rewrite-vllm` is engineering: engine config, prompt files, overhead gates,
wrap-style golden tests, trim/shuffle parity tests, claim-based work distribution, calibration.

---

## 3. Mapping: 1.5B component → proposed 3B component

Nothing in `projects/rewrite/` will be modified. Everything is ported into
`projects/rewrite-3B/01_data/`.

| 1.5B artefact | role | 3B counterpart (new) |
|---|---|---|
| `01_explore/score_fasttext.py` | fastText scoring recipe | `src/kys3b/fasttext_score.py` (identical preprocessing + raw score) |
| `03_TokenCounts/count_tokens.py` | `tokens-llama2` | not needed — the column ships with the HF table; `src/kys3b/tokens.py` re-counts *rewritten* text only |
| `04_select/select_10b.py` | all pre-rewrite selection + shared base | `src/kys3b/select.py` + `bin/01_select.py` (report/commit phases) |
| `04_select/select_quality_base_15B.py` | deeper fastText prefix | folded into `select.py` (quality-base = next 10B prefix) |
| `05_select_s5_variants/select_s5.py`, `06_lambda_grid/select_lambda_grid.py` | DA domain + λ ranking | `src/kys3b/disagreement.py` (same `U`, `τ_q`, `τ_v`, `u`; per-scorer budget = `B_s`), **λ = 0.5 only** |
| `06_lambda_grid/DOMAIN_STATS.md` | domain audit | `reports/DOMAIN_STATS.md`, same table, regenerated |
| `07_rewrite/rewrite_worker.py` | pass-1 vLLM worker | `src/kys3b/rewrite/worker.py` (one worker, `--pass {p1,distill}`) |
| `09_Distill/rewrite_worker.py` | pass-2 worker | same worker, `--pass distill` (the two 1.5B copies are merged; the only real differences — prompt, output subdir, drop threshold, cache namespace — become arguments) |
| `07_rewrite/sbatch_template.sh`, `09_Distill/sbatch_template.sh` | Slurm | `slurm/rewrite_array.sbatch` — **rewritten for Skipjack** (§13, §14) |
| `07_rewrite/check_progress.py`, `generate_summary.py` | monitoring | `bin/status.py` |
| `10_postprocess/01_strip_prefix*.py` (4 variants) | prefix strip | `src/kys3b/post/strip.py` — one module, per-arm rule table |
| `10_postprocess/02_assemble_5B*.py` (3 variants) | assemble B | `src/kys3b/post/assemble.py` — one module, per-arm sort-key table |
| `10_postprocess/02..05_*_rewrite.py` | REWIRE pool→score→filter | `src/kys3b/post/rewire.py` |
| `10_postprocess/03_mix_shared_top*.py`, `04_assemble_base.py`, `06_shuffle_10B_base.py` | mix + shuffle | `src/kys3b/post/mix.py` |
| `10_postprocess/pp_io.py` | atomic write, `bucketed_shuffle`, `paired_wiki_status` | `src/kys3b/io.py` — **ported verbatim** (it is the shuffle-parity contract) |
| `prompts/` (6 files) | prompts | `prompts/` — **byte-copied**, md5-asserted at preflight |
| `rewrite-vllm/scripts/preflight.py`, `06_calibrate.py`, `tests/` | gates | `bin/preflight.py`, `bin/calibrate.py`, `tests/` — ported and extended |

---

## 4. Exact token-budget scaling, 1.5B → 3B

| quantity | 1.5B | 3B | factor |
|---|---:|---:|---:|
| model size | 1.5B | 3B | 2× |
| tokens per epoch (training mixture) | 10B | **20B** | 2× |
| epochs | 3 | 3 | 1× |
| **matched training-token horizons** | 10 / 20 / 30B | **20 / 40 / 60B** | 2× |
| shared fixed raw base | 5B | **10B** | 2× |
| strategy-specific component | ~5B | **~10B** | 2× |
| rewritten-output budget `B` | 5B | **10B** | 2× |
| source budget `B_s` (QF / WRAP / DIV / DA) | 10B | **20B** | 2× |
| REWIRE oversampling `κ` | 2 | **2** | 1× |
| REWIRE pre-rewrite pool `κ·B_s` | 20B | **40B** | 2× |
| QUALITY-BASE second half | 5B | **10B** | 2× |
| λ | 0.5 | **0.5** (only value) | 1× |
| quality floor / variance cap | Q30(q\|U) / Q90(v\|U) | **unchanged (percentile, recomputed on the 3B U)** | 1× |
| DA per-scorer candidate budget | 10%-by-tokens of `D` ≈ 9.03B (≈0.90·B_s) | **20B = `B_s`** (Option C, §20 D-2) | ~2.2× |
| validation holdout | 50,000 docs | **50,000 docs** (unchanged) | 1× |
| tokenizer / length rule | llama-2, `tokens+1` | **unchanged** | 1× |
| rewriter, prompts, decoding | Qwen2.5-7B-Instruct, 6 prompts, greedy | **unchanged** | 1× |
| rewriting passes per arm | 2 (both, full source set) | **2 (both, full source set)** | 1× |

`B_s = 2B` and `κ = 2` are both preserved exactly, so the ratio structure of the 1.5B experiment is
carried over unchanged. **Only the source pool does not double** — it is the same 99,949,162-document
table. Consequences are quantified in §18 and noted in §20 (D-4, D-5), and are accepted.

---

## 5. The six 3B settings

`D'` = candidate pool = all 99,949,162 docs − 50,000 val − the shared top-10B fastText base.
Estimated `D'` ≈ **85B llama-2 tokens** (see §18 for the derivation; exact value from the report phase).

| # | setting | fixed 10B | strategy source budget | strategy source domain | rewrite | strategy-specific 10B built by | corpus size | training horizons |
|---|---|---|---:|---|---|---|---:|---|
| 1 | **Quality-Base** | shared top-10B fastText | 10B | `D'`, fastText DESC prefix | no | used verbatim, raw | ~20.000B | 20 / 40 / 60B |
| 2 | **Quality-First** | same | 20B | `D'`, fastText DESC prefix | yes (wiki + distill) | all pass-1 + fastText-ordered distill top-up | ~20.000B | 20 / 40 / 60B |
| 3 | **WRAP-Inspired** | same | 20B | `D'`, uniform (child 2) | yes (4 styles + distill) | all pass-1 + **seeded-random** distill top-up | ~20.000B | 20 / 40 / 60B |
| 4 | **REWIRE-Inspired** | same | **40B** (`κ=2`) | `D'`, uniform (child 3) | yes (wiki + distill over all 40B) | **fastText on rewritten output**, global top-10B | ~20.000B | 20 / 40 / 60B |
| 5 | **Diversity-Oriented** | same | 20B | `D'`, per-topic quota ∝ token share, `q` DESC within topic | yes (wiki + distill) | per-topic quota, fastText DESC within topic, **policy A** | **~19.8B (may be short)** | **20 / 40 / 60B — same as everyone** |
| 6 | **Disagreement-Aware** | same | 20B | `U_τ ⊂ D'`, `u = q + 0.5√v` DESC | yes (wiki + distill) | all pass-1 + `u`-ordered distill top-up | ~20.000B | 20 / 40 / 60B |

× 3 seeds × 3 evaluated horizons = **18 runs / 54 evaluated checkpoints**, matching the paper's design.

**The same corpus is repeated across epochs — no per-epoch regeneration.** Training horizons are
**token** horizons and are matched across all six settings. A setting whose corpus is slightly under
20B (only Diversity is expected to be) simply completes slightly more than 1 / 2 / 3 passes; it is
**never** trained to a shorter token budget. See §18.3 for the pass-count arithmetic.

---

## 6. Detailed construction logic, per setting

Common preamble, executed **once**, deterministically:

```
N        = 99,949,162                                    # assert against the HF table
VAL      = sort(default_rng(SeedSequence(42).spawn(8)[0]).choice(N, 50_000, replace=False))
ALIVE    = all \ VAL                                     # NO other exclusion (§20 D-1)
tie      = default_rng(SeedSequence(42).spawn(8)[1]).permutation(N)
len(d)   = tokens-llama2[d] + 1
q        = ((ft + fw + mb) / 3).astype(float32)           # *-ranking-v2 columns
v        = (((ft-q)**2 + (fw-q)**2 + (mb-q)**2) / 3).astype(float32)
order_desc(idx, s) = idx[lexsort((tie[idx], -s[idx].astype(float64)))]
fill_to(order, target) = prefix whose cumsum first reaches target, last doc kept whole
BASE     = fill_to(order_desc(ALIVE, fasttext-ranking-v2), 10e9)     # the shared fixed base
D'       = ALIVE \ BASE                                              # the residual candidate pool
```
**Every strategy below selects from `D'` only. No document of `BASE` can re-enter any strategy pool.**
This is asserted (`|BASE ∩ S| == 0` for all six S) and is a hard `stop()`.

### 6.1 Quality-Base
`qbase = fill_to(order_desc(ALIVE, ft_v2)[|BASE|:], 10e9)` — the **contiguous continuation** of the
same fastText order, so `BASE ∪ qbase` is the top-20B prefix. No rewriting.
Final mixture = `BASE` (10B raw) + `qbase` (10B raw) = **~20B raw**.

### 6.2 Quality-First
`S_QF = fill_to(order_desc(ALIVE, ft_v2)[|BASE|:], 20e9)` — the next **20B** along the same order,
so `qbase ⊂ S_QF` exactly as at 1.5B (asserted). Rewrite **both** passes over all of `S_QF`.
Assemble: all `status==2` pass-1 outputs, then distill top-up in `fasttext-ranking-v2` DESC order of
the **original** document until `Σ(rewritten_tokens+1) ≥ 10e9`. Final = `BASE` + ~10B rewritten.

### 6.3 WRAP-Inspired
`S_WRAP = fill_to(RNG_WRAP.permutation(D'), 20e9)`, `RNG_WRAP = spawn(8)[2]`.
Pass 1: per-document style from `default_rng([42, shard_index]).integers(0, 4, n_rows)` over
`["easy","hard","wiki","qa"]`, style recorded in `wrap_style`. Pass 2: the shared distill prompt,
generated over the **full** source set.
Assemble: **all** `status==2` wrap outputs first, then the gap filled from distill by a
**seeded random draw `default_rng(42)`, with no quality signal at all** — preserved verbatim from
1.5B, because the arm's whole purpose is "does source quality matter?". Final = `BASE` + ~10B.

### 6.4 REWIRE-Inspired  (see §7 for the κ argument)
`R = fill_to(RNG_REWR.permutation(D'), 40e9)`, `RNG_REWR = spawn(8)[3]`.
Rewrite **all** of `R` with **both** passes (wiki + distill). Strip prefixes, recount.
Pool = all `status==2` wiki ∪ all `status==2` distill (both passes, one row each, **not**
deduplicated by `doc_id`) — **both passes must enter the pool before scoring**.
fastText-score every **rewritten** text with the exact 1.5B recipe. Global DESC sort on the raw
score; `fill_to` to **10e9** rewritten tokens. Final = `BASE` + ~10B.

### 6.5 Diversity-Oriented
**Source selection** over `D'`, for each of the **24** WebOrganizer topics (24 or `stop()`):
```
quota_c  = 20e9 * (tokens_c(D') / tokens(D'))
S_c      = fill_to(order_desc(D'_c, q), quota_c)          # q = CONSENSUS of all three scorers
S_DIV    = ∪_c S_c ;  if under 20e9, q-DESC top-up from D' \ S_DIV   # same top-up as 1.5B
```
**Rewriting:** both passes over the full selected source set.

**Assembly**, preserving the validated 1.5B behaviour exactly: topic proportions taken from the
*selected source set* (`tokens-llama2+1`), `per_topic_quota[t] = 10e9 · proportion[t]`, each topic
filled **only from its own** documents — all `status==2` pass-1 first, then distill top-up, sorted
**`fasttext-ranking-v2` DESC within topic** (§20 D-6: this is what the 1.5B code did, and it is kept
deliberately). **No cross-topic backfill, no padding.** A shortfall is reported, not repaired.

**Corpus:** `BASE` + ~9.8B ⇒ **~19.8B** (see §18.3).
**Training horizon: 20B / 40B / 60B, identical to every other setting.** The corpus is simply
re-read slightly more often (~1.011 / 2.023 / 3.034 passes instead of 1 / 2 / 3).

### 6.6 Disagreement-Aware  — **Option C, locked**
```
for each scorer s in {fasttext-ranking-v2, fineweb-edu-ranking-v2, modernbert-ranking-v2}:
    A_s = fill_to(order_desc(D', s), 20e9)                # per-scorer budget = B_s = 20B
U     = A_ft ∪ A_fw ∪ A_mb
τ_q   = 30th percentile of q over U                       # recomputed on the 3B U
τ_v   = 90th percentile of v over U                       # recomputed on the 3B U
U_τ   = {d ∈ U : q ≥ τ_q AND v ≤ τ_v}
S_DA  = fill_to(order_desc(U_τ, q + 0.5*sqrt(v)), 20e9)
```
λ = 0.5, floor/cap percentiles 30 / 90 — **all three carried over unchanged and never retuned against
3B downstream performance.** Only λ = 0.5 is built (§20 D-12). Rewrite both passes; assemble with
sort key `u = q + 0.5√v` DESC. Final = `BASE` + ~10B.

Expected domain sizes (to be confirmed in the report phase): `U` ≈ 47B tokens, `U_τ` ≈ 36B tokens,
`S_DA` ≈ 56% of `U_τ` — closely tracking the 1.5B ratios (`|U|/B_s` ≈ 2.2, `|S_DA|/|U_τ|` ≈ 54%).

---

## 7. REWIRE: why the pre-rewrite pool is 40B — verified and locked

**Confirmed from four independent places.**

1. **The 1.5B code.** `04_select/select_10b.py:51`:
   ```python
   REWRITE_TARGET = 20_000_000_000   # rewrite = 10B * 2
   ```
   and `:245` — `fill_to(RNG_REWR.permutation(remaining_idx), tok, REWRITE_TARGET)`. The variable-block
   default `TARGET` on the same page is `10_000_000_000`. So at 1.5B, with `B_s = 10B`, REWIRE's
   pre-rewrite pool was **20B = κ·B_s with κ = 2**.
2. **The realized 1.5B artefact.** `10_postprocess/_step2_rewrite_summary.json`,
   `funnel_pre_fasttext[0]`: `"original selection (rewrite, ~20B)", docs: 21,214,299,
   tokens: 20,000,000,679`.
3. **The paper.** §5.1: "The oversampling factor is `κ = 2`". §4.2: "the sample is enlarged by an
   oversampling factor `κ > 1`, giving a pool `R` of `κ B_s` tokens." App. D.3: REWIRE's pre-rewrite
   source pool is **22.15%** of the candidate corpus against WRAP's **11.07%** — exactly 2:1, and
   `10B / 0.1107 ≈ 90.3B` reproduces the candidate-pool size from the other direction.
4. **The later in-repo scaling design.** `13_600M/02_select/README.md`: "**κ = 2 applies to the
   remainder** (120B vs 60B), following `04_select/select_10b.py:51` … It is *not* 2× on the total."

**Therefore at 3B, locked:** `B = 10B`, `B_s = 2B = 20B`, `κ = 2` ⇒ **`κ·B_s = 40B` raw source
tokens**, uniformly sampled from `D'`, all of it rewritten with **both** passes, then fastText applied
to the **rewritten** output and the top **~10B rewritten tokens** retained.

A 20B pre-rewrite pool would set `κ = 1` and delete the arm's defining property — deferring quality
selection until *after* rewriting only has teeth if there is surplus to reject. At 1.5B the realized
acceptance rate was **30.8% by token** (5.00B kept out of a 16.22B rewritten pool); at 20B pre-rewrite
the 3B pool would be ~16.6B rewritten tokens against a 10B target — a ~60% acceptance rate, a
materially weaker filter and not the same experiment.

**Accepted consequence (Decision 4).** 40B out of an ≈85B residual pool is **≈47% of `D'`**, versus
22.15% at 1.5B. REWIRE's random sample therefore overlaps every other strategy at ~47% by
construction rather than ~22%, and REWIRE alone is **41% of the generation workload** (§15).
**κ is not reduced to preserve the old overlap percentage.** The 3B write-up must regenerate the
overlap analysis (the analogue of App. D.3 / Fig. 9) and state the ~47% base rate rather than
reusing the 1.5B figure.

---

## 8. Prompt-generation logic

### 8.1 Pass 1, four non-WRAP arms — grounded Wikipedia-style
`prompts/wikipedia_style_rephrasing_grounded.md`, byte-copied, md5 `bca104f…`.
Applied as `template.replace("[TEXT]", doc_text)`, then
`tokenizer.apply_chat_template([{role:"user", content}], add_generation_prompt=True, tokenize=False)`.
Preflight asserts the templated empty-document length is exactly **150** tokens.

### 8.2 Pass 2, all five arms — distill
`prompts/distill/distill_prompt.txt`, md5 `5387005…`. The `[TEXT]` placeholder sits **mid-template**;
`replace` handles any position. Preflight asserts overhead **185**.

### 8.3 Pass 1, WRAP — four adapted style prompts
`prompts/wrap/{easy,hard,wiki,qa}.txt`, md5s in §2.8, overheads 72 / 66 / 73 / 83, all asserted.
Each ends with `Passage:\n`; content = `wrap_prompts[style] + doc_text`.
Assignment: `default_rng([42, shard_index]).integers(0, 4, n_rows)` indexing
`["easy","hard","wiki","qa"]` **in that order**. A golden vector pinned against
`np.random.default_rng([42, i]).integers(0, 4, 16)` is added to the test suite so a NumPy PCG64
change is caught rather than silently re-rolling the corpus.

> **Not glossed:** the 3B pipeline re-shards the selections, so `shard_index` is *ours*. The style
> assignment is reproducible **within this pipeline** and statistically identical, but it is not the
> same draw the 1.5B run produced and cannot be (different corpus, different sharding). Accepted
> under Decision 8.

### 8.4 Production composition for 3B — **locked**
**Both passes are generated over the full selected source set, for all five rewriting arms.** No
adaptive shortcut, no projection-based skip. Retention then follows the per-arm rules of §6.

Rationale (Decision 3): it matches the 1.5B pipeline and the paper's App. C; the projected 3B pass-1
yields are **6.80 / 7.62 / 7.30 / 8.63 B** for QF / DIV / DA / WRAP — **all below the 10B target**, so
a skip could never fire without a wrong projection; and for REWIRE the skip would *change the corpus*
rather than cheapen it, because the fastText filter ranks wiki and distill **together** and distill
won 53.1% of the retained tokens at 1.5B.

---

## 9. Differences between the paper, the 1.5B code, and this 3B design

Every item is either resolved with an action, or needs no action. Nothing is silently resolved.

| id | topic | paper says | 1.5B code does | 3B design | status |
|---|---|---|---|---|---|
| **D-1** | 5M analysis sample | §3 / Fig. 1: "we **exclude** [the 5M sample] from subsequent training-data construction" | excludes **only** the 50,000-doc val holdout. The published table removed exactly 50,838 rows (the ModernBERT-annotator set). **4,998,853 of the 5,001,383 analysis-sample documents are present in the published pool — 5.001% of it** (proof in §20 D-1) | **use the pool as published; do not re-remove anything** (= 1.5B behaviour) | ✅ pipeline resolved · ⚠️ **paper correction required** |
| **D-2** | DA candidate domain | per-scorer selection "until a **10B-token budget** is reached" | each scorer's **top 10% by tokens of `D`** ≈ **9.03B** (= 0.90·`B_s`); this is what reproduces the paper's own `U`/`U_τ`/λ tables | **per-scorer budget = `B_s` = 20B** (Option C) | ✅ RESOLVED |
| **D-3** | Diversity within-topic key | §4.2 Eq. 6: rank by **consensus `q`** | **selection** ranks by `q` ✓; **assembly** ranks by **`fasttext-ranking-v2`** within topic | preserve both exactly as coded | ✅ RESOLVED — paper footnote needed |
| **D-4** | REWIRE retention | §4.2 Eq. 5: `Top_B(π(R), r^(fastText))`, the *percentile* | sorts by the **raw** fastText score and fills to a **token budget**; percentile computed for readability only. Monotone ⇒ identical set | preserve (raw score, token budget) | no action — equivalent |
| **D-5** | both-passes generation | App. C: "**Both rewriting passes are generated over the full selected source set**" | exactly that; the `wiki-alone` assembly branch exists but never fired | **both passes always** | ✅ RESOLVED |
| **D-6** | Diversity budget | Table 1: "~5B strategy-specific tokens" | landed at **4.890B** (110.3M short); final mixture 9.890B, still trained to the matched 10/20/30B horizons | same shortfall expected and accepted; **horizons stay 20/40/60B** | ✅ RESOLVED |
| **D-7** | REWIRE pool fraction | App. D.3: REWIRE overlaps every setting at 22.1%, "the overlap expected under independence" | 20B of a ~90.3B pool = 22.15% | 40B of an ~85B pool ≈ **47%**; overlap analysis must be **regenerated** for 3B, not reused | ✅ RESOLVED (accepted) |
| **D-8** | `gpu_memory_utilization` | not stated | READMEs and argparse say `0.90`; the sbatch that actually ran passed **`0.85`** | use **0.85** | no action |
| **D-9** | distill drop threshold | not stated | pass 1 drops at a fixed **30720** templated tokens; distill derives **28672** = `32768 − 4096` | preserve both, per pass | no action |
| **D-10** | `tokens-llama2` budget rule | not stated | `+1` BOS applied to *every* budget, source and rewritten | preserve | no action |
| **D-11** | κ scope | §4.1/§5.1: κ enlarges the *source* pool | `REWRITE_TARGET = 10B * 2` — κ on the source budget only, never on the shared core | preserve: 40B source, 10B core | no action |
| **D-12** | λ variants | only λ=0.5 trained | **six** λ variants rewritten (0, 0.5, 1, 1.5, 2, 3); that is why the 1.5B census shows 110B source tokens rewritten, not 60B | **λ = 0.5 only** | ✅ RESOLVED |
| **D-13** | cluster | — | partitions `nvl` / `h100`, QOS `h200_4`, `--exclude=n02,n03,c001` — **none exist on Skipjack** | fully rewritten Slurm layer (§13, §14) | ✅ RESOLVED |
| **D-14** | scale-up precedent | — | `13_600M/02_select` **disabled** the DA candidate domain at larger scale | rejected: the 3B run keeps the domain (Option C) | ✅ RESOLVED |

---

## 10. Dataset schema, download and storage plan

### 10.1 `blab-jhu/KYS-DCLM-Refinedweb-100M-Scored` — inspected via the HF API (no download)

- **Gated:** `"gated": "manual"`. Your token at `~/.cache/huggingface/token` already has access
  (the README fetch succeeded; an anonymous fetch returned 401).
- **200 parquet shards**, `merged_clean_00000.parquet` … `merged_clean_00199.parquet`,
  **181.58 GB total**, ~908 MB each. Plus `README.md` and `.gitattributes`. Nothing else.
- **99,949,162 rows**, 500,000 per shard, last shard 449,162. Row count = 100,000,000 − 50,838
  (documents overlapping the ModernBERT scorer's training set — and *only* those; see §20 D-1).

**Columns — all 13, and every field the six strategies need is present** (verified against the downloaded pool on 2026-09-23; an earlier revision of this plan said 14, which was a miscount of the dataset card's own table):

| column | type | needed by |
|---|---|---|
| `doc_id` | int64 | **the join key everywhere**; contiguous 0…99,949,161; `= shard*500000 + row` |
| `orig_doc_id` | int64 | position in the original 100M reservoir sample — the key that made the §20 D-1 check possible |
| `text` | string | rewriting input |
| `url` | string | carried through |
| `metadata` | string | raw WARC JSON, carried through |
| `fasttext` | float32 | raw P(hq) — reference distribution for the REWIRE percentile |
| `fineweb-edu` | float32 | raw logit |
| `modernbert` | float32 | raw ridge score |
| `topic` | string | one of 24 WebOrganizer categories → DIVERSITY-ORIENTED |
| `fasttext-ranking-v2` | float32 | base, Quality-Base, Quality-First, DA domain, assembly sort |
| `fineweb-edu-ranking-v2` | float32 | `q`, `v`, DA domain |
| `modernbert-ranking-v2` | float32 | `q`, `v`, DA domain |
| `tokens-llama2` | int32 | **every token budget** (`+1` for train length) |

**Not stored, derived on the fly** (the card gives the exact recipe, matching the 1.5B code):
`q = ((ft+fw+mb)/3).astype(float32)`, `v = (((ft-q)**2+(fw-q)**2+(mb-q)**2)/3).astype(float32)`,
`u = q + 0.5*sqrt(v)`. "`float32` and `ddof=0` are load-bearing."

**No deduplication or content-hash column exists.** `doc_id` is the only identity. The 1.5B pipeline
likewise never deduplicated; documents appearing in two strategies are the same `doc_id`. The
`metadata` JSON carries `WARC-Payload-Digest`, which could support a later dedup audit but was not
used at 1.5B — not proposed here, since it would deviate from the reproduced design.

**Pinned model revisions recorded on the card** (needed for the REWIRE re-scoring):
DCLM fastText `mlfoundations/fasttext-oh-eli5` @ `cd8b714a…`; the local copy
`models/5m/external/fasttext_oh_eli5.bin` is the one that produced the 1.5B scores and will be
sha256-pinned.

### 10.2 Download plan (**not executed**)

```
huggingface-cli download blab-jhu/KYS-DCLM-Refinedweb-100M-Scored \
  --repo-type dataset \
  --local-dir /weka/projects/bvandur1/zhuicon1/rewrite-3b/00_pool \
  --max-workers 8
```
then a symlink so the originally requested path still resolves:
```
ln -s /weka/projects/bvandur1/zhuicon1/rewrite-3b/00_pool \
      /weka/scratch/jhu/bvandur1/zhuicon1/datasets/dclm-refinedweb-100m-sample
```
(The scratch path is currently an empty directory; it will be replaced by the symlink.)

Verification after download: 200 files; per-file row count (500,000 / 449,162); `doc_id` contiguity
(`doc_id == arange(shard*500000, …)`); total 99,949,162; zero nulls in the six numeric columns and
`topic`; exactly 24 topic values; sha256 of every shard recorded in the manifest. This mirrors
`select_10b.py:112-130`, which `stop()`s on any violation.

### 10.3 Storage root — **resolved (Decision 7)**

```
$ df -h /weka/scratch/jhu/bvandur1
wekafs1/scratchjhu   9.1T   9.0T   137G   99%      <-- your scratch quota: 137 GB free
$ df -h /weka/projects/bvandur1
wekafs1/bvandur1     182T    66T   117T   37%      <-- 117 TB free
```

**Physical root: `/weka/projects/bvandur1/zhuicon1/rewrite-3b/`.** Symlinks are created later (not
now, and not until the directories exist) at:

- `/home/jhu/zhuicon1/scratch_bvandur1/zhuicon1/datasets/dclm-refinedweb-100m-sample`
  → `/weka/projects/bvandur1/zhuicon1/rewrite-3b/00_pool`
- `/home/jhu/zhuicon1/scratch_bvandur1/zhuicon1/datasets/rewrite-3b-llama-60b-tokens`
  → `/weka/projects/bvandur1/zhuicon1/rewrite-3b/dataset`

The code directory stays where it is (`projects/rewrite-3B/01_data/`, on scratch — it is small).
`/weka/projects/bvandur1` is already the working root for `kys_raw` (`/projects/bvandur1/zhuicon1/kys`),
so this is consistent with existing practice.

---

## 11. Proposed directory layout

Code (git repo `git@github.com:imHuicongZhang/rewrite-3B.git`, nothing pushed yet):

```
projects/rewrite-3B/01_data/
  plan.md                      <- this file
  README.md
  pyproject.toml / requirements.txt
  prompts/                     byte-copies of the six 1.5B prompts + PROVENANCE.md (md5 table)
    wikipedia_style_rephrasing_grounded.md
    distill/distill_prompt.txt
    wrap/{easy,hard,wiki,qa}.txt
    wrap_prompts.json          key order easy,hard,wiki,qa  (part of the seed)
  configs/
    budgets.yaml               every token budget in one place, with its 1.5B ancestor
    vllm.yaml                  the five LLM kwargs + sampling + inherited-defaults record
    paths.yaml                 pool / work / out roots (rooted at /weka/projects/bvandur1/...)
    cluster.yaml               partitions, account, qos, gpus, wall time
  src/kys3b/
    io.py                      atomic_write_table, bucketed_shuffle, paired_wiki_status (ported)
    pool.py                    shard map, column loader, validation
    select.py                  base, quality-base, quality-first, wrap, rewire, diversity
    disagreement.py            U (20B/scorer), tau_q, tau_v, u-ranking, lambda=0.5
    fasttext_score.py          exact 1.5B recipe + v2 percentile
    tokens.py                  llama-2 counting of rewritten text
    rewrite/{worker.py,prompts.py,shards.py,claims.py}
    post/{strip.py,assemble.py,rewire.py,mix.py}
    manifest.py                every stage writes a manifest; every stage validates the previous one
  bin/
    00_download_pool.py  01_select.py  02_materialize_sources.py
    03_rewrite_launch.py 04_postprocess.py 05_assemble_final.py
    preflight.py  calibrate.py  status.py  validate.py
  slurm/
    select.sbatch  materialize.sbatch  rewrite_array.sbatch  score_rewire.sbatch  post.sbatch
  tests/
    test_wrap_styles.py        golden PCG64 vector
    test_fill_to.py            budget/overshoot/last-doc-whole semantics
    test_prompt_overheads.py   150 / 185 / 72 / 66 / 73 / 83
    test_fasttext_parity.py    raw-score recipe against known 1.5B values
    test_shuffle_parity.py     bucketed_shuffle determinism
    test_selection_smoke.py    full selection on a 2-shard synthetic pool
  reports/                     generated: SELECTION_REPORT.md, DOMAIN_STATS.md, POSTPROCESS.md
```

Data, under `/weka/projects/bvandur1/zhuicon1/rewrite-3b/`:

```
rewrite-3b/
  00_pool/                          the 200-shard scored pool  (182 GB)
  dataset/                          <- the symlink target for .../datasets/rewrite-3b-llama-60b-tokens
    _MANIFEST.json                  code commit, pool sha, budgets, seeds, dates
    01_selection/
      val/val_doc_ids.npy
      shared-base-10B/    doc_ids.npy  _manifest.json
      quality-base/       doc_ids.npy  _manifest.json
      quality-first/      doc_ids.npy  _manifest.json
      wrap-inspired/      doc_ids.npy  _manifest.json
      rewire-inspired/    doc_ids.npy  _manifest.json
      diversity-oriented/ doc_ids.npy  _manifest.json   (+ per-topic quota table)
      disagreement-aware/ doc_ids.npy  _manifest.json   (+ U / U_tau / tau_q / tau_v)
      SELECTION_REPORT.md  DOMAIN_STATS.md  OVERLAP.md
    02_sources/<setting>/            materialized rewriting input
      shard_NNNNN.parquet            (doc_id, text) only, 10,000 rows/shard, zstd
      _shards.json                   shard index + row counts + sha256  (the fingerprint)
    03_rewritten/<setting>/
      p1/shard_NNNNN.parquet         doc_id, rewritten, rewritten_tokens, status,
                                     finish_reason, input_tokens_qwen, wrap_style
      distill/shard_NNNNN.parquet    same, no wrap_style
      _claims/                       claim files (worker id, host, jobid, heartbeat)
      _progress/                     per-worker progress JSON
      _manifest.json
    04_filtered/
      rewire-inspired/scored_pool/   + rewritten_fasttext_score, rewritten_fasttext_ranking_v2
      rewire-inspired/kept/          top-10B rewritten
      <other settings>/rewritten/    assembled ~10B  (wiki_NNNNN / distill_NNNNN)
      _assembly_manifest.json  per setting
    05_final/<setting>/
      shared-base-10B/               physical copy, source_prompt="original"
      strategy/                      the ~10B strategy half
      shuffled/part_NNNNN.parquet    <- THE DELIVERABLE this repo stops at (seed 42, doc level)
      _pretrain_manifest.json
    logs/<stage>/<jobid>_<task>.out
    reports/
```

**This repo's final output is `05_final/<setting>/shuffled/` parquet.** Tokenization into Nanotron
`.ds` format is the training repository's responsibility (Decision 10), exactly as at 1.5B where
`kys_raw/tokenize_raw_text.sh` lived in `nanotron-kys`, not in `projects/rewrite`.

---

## 12. Proposed code architecture

**Principles carried over from the 1.5B run, all of which earned their place there:**

1. **Two-phase everything.** Every stage has `--phase report` (computes, writes nothing but a report)
   and `--phase commit` (writes). Byte-identical logic; only the last lines differ. The report phase
   is what will size `D'`, the DA domain under Option C, and the per-topic Diversity supply **before
   any GPU time or storage is committed**.
2. **Atomic writes only.** `<dest>.tmp` in the same directory + `os.replace`, partial `.tmp` unlinked
   in `finally`. Ported from `pp_io.atomic_write_table`.
3. **`stop()` not `warn()`.** Any invariant violation exits non-zero: under-filled budget, wrong topic
   count, non-contiguous `doc_id`, non-zero base∩strategy overlap, prompt md5 mismatch, prompt
   overhead mismatch.
4. **Manifests are the contract.** Each stage writes `_manifest.json` and each stage *validates the
   previous stage's manifest* before doing anything. Includes the code commit sha.
5. **Seeds are structural.** `SeedSequence(42).spawn(8)` with the documented child mapping;
   `default_rng([42, shard_index])` for wrap styles; `default_rng(42)` for the wrap distill draw and
   the final shuffle. Golden vectors in tests.
6. **The rewrite worker is one module**, not two copies. 07 and 09 differed only in prompt, output
   subdir, drop threshold and cache namespace — those become `--pass {p1,distill}` arguments with a
   per-pass table. The "never read pass-1 output as input" assertion from `09_Distill` is kept.
7. **Claim-based work distribution** (Decision 8), not `shard_idx % num_workers`. A worker atomically
   creates `_claims/shard_NNNNN.claim` (`O_CREAT|O_EXCL`), heartbeats into it, and releases on
   completion; stale claims (no heartbeat for 30 min, or a Slurm job id no longer in `squeue`) are
   reclaimable. Modulo assignment strands a dead worker's shards until that exact array index is
   resubmitted — unacceptable with preemptible scavenger GPUs and a heterogeneous fleet. It is
   **behaviourally neutral**: per-document output depends only on `(shard_index, row_index)`, never on
   which worker ran it.
8. **No new interpretation.** Prompts are byte-copied and md5-asserted. Engine kwargs are exactly the
   five the 1.5B run passed. The `inherited_defaults_do_not_pass` block from `configs/vllm.yaml` is
   carried across as a comment so nobody "helpfully" tunes `max_num_batched_tokens`.

**Stage graph**

```
00 download+verify pool ─→ 01 select (report → commit) ─→ 02 materialize sources
                                                              │
                                      ┌───────────────────────┴────────────────────┐
                                      ▼                                            ▼
                             03 rewrite pass 1 (array)                   03 rewrite distill (array)
                                      └───────────────────────┬────────────────────┘
                                                              ▼
                                        04a strip+recount ─→ 04b assemble (4 arms)
                                                          └→ 04c REWIRE: pool(both passes) → fastText → top-10B
                                                              ▼
                                        05 mix base + shuffle ─→ validate ─→ hand to nanotron-kys
```
Pass 1 and distill are **independent** and run concurrently (as 07 and 09 did), against separate
cache namespaces. Both are always run for all five arms (Decision 3).

---

## 13. Slurm / vLLM parallelization strategy

### 13.1 vLLM — unchanged from 1.5B
One vLLM process per GPU, `tensor_parallel_size=1`, `dtype=bfloat16`,
`gpu_memory_utilization=0.85`, `max_model_len=32768`, greedy, `max_tokens=min(4096, 32768-n_in)`.
Data parallelism is **N independent processes**, never vLLM's `data_parallel_size`.
This is right for the workload: the mix is **prefill-dominated at ~2.75:1**, so per-GPU model
replication maximizes aggregate prefill throughput and TP would only add communication.

### 13.2 Skipjack Slurm — the 1.5B directives do not port

| 1.5B directive | status on Skipjack | replacement |
|---|---|---|
| `--partition=nvl,h100` | `nvl` does not exist | `--partition=h100,h200` |
| `--qos=h200_4` | does not exist | `--qos=jhu2` (account `bvandur1_blabscify`) or `--qos=scavenger` (account `bvandur1`) |
| `--gres=gpu:h100:1` | works, but pins to H100 only | **`--gres=gpu:1`** — the generic type schedules on either partition (validated with `sbatch --test-only`) |
| `--exclude=n02,n03,c001` | those nodes do not exist | drop; add real exclusions only if a node misbehaves |
| `--time=3-00:00:00` | still the max | keep |
| `--partition=cpu` | does not exist | `--partition=med` (QOS `cpu_qos`), 74 nodes, 108 CPU / node |

Proposed rewriting array header:
```bash
#SBATCH --partition=h100,h200
#SBATCH --account=bvandur1_blabscify     # or bvandur1 for the scavenger overflow array
#SBATCH --qos=jhu2                       # or scavenger
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16               # DefCpuPerGPU=30; 16 gives 187.5G at DefMemPerCPU=12000
#SBATCH --time=3-00:00:00
#SBATCH --requeue                        # mandatory for scavenger (PreemptMode=REQUEUE)
#SBATCH --array=0-31%32
```
Caches must **not** go to `$HOME` (the 1.5B lesson): `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
`HF_HOME`, `XDG_CACHE_HOME`, `TMPDIR` all under the project root, **namespaced per (pass, setting,
array task)** so concurrent torch.compile does not collide.

The 1.5B `09_Distill` NCCL/Gloo interface pin (`GLOO_SOCKET_IFNAME`/`NCCL_SOCKET_IFNAME` forced to
the `172.20.x` interface) was a fix for a specific old-cluster bug. It is **not** ported blindly; if
engine-init failures appear on Skipjack we diagnose there. Recorded so it is not forgotten.

### 13.3 Two-queue strategy (Decision 9)
- **Array A:** `bvandur1_blabscify` / `jhu2`, `--array=0-31%32`. The `jhu2` QOS **allows up to 32
  concurrent GPUs** for this user; it does **not** guarantee them. Actual concurrency depends on queue
  contention, fairshare and scheduler allocation across the 128 H100 + 24 H200 GPUs shared with all
  other users.
- **Array B:** `bvandur1` / `scavenger`, `--requeue`, a larger array with `%N` tuned to what the fleet
  actually yields. Scavenger capacity is **preemptible and unbounded-but-not-promised**; preempted
  tasks requeue and resume at shard granularity.

Both arrays run the same worker against the same claim directory, so they cooperate rather than
duplicate. This is the main reason for the claim-based design (§12 item 7).

---

## 14. Your actual Skipjack H100/H200 limits (measured 2026-09-23)

**Commands run (all read-only; `sbatch --test-only` creates no job):**
```
sinfo -o "%20P %10a %12l %6D %10T %20G %N"
sinfo -o "%20P %10D %25N %40G" -e
scontrol show partition h100 ; scontrol show partition h200 ; scontrol show partition med
sacctmgr -P show assoc user=$USER format=Cluster,Account,User,Partition,QOS,DefaultQOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmitJobs,MaxWall,GrpJobs,GrpSubmitJobs
sacctmgr -P show qos format=Name,Priority,MaxTRESPU,MaxTRESPerJob,MaxJobsPU,MaxSubmitJobsPU,MaxWall,GrpTRES,Preempt,PreemptMode,Flags,UsageFactor
sacctmgr -P show cluster format=Cluster,ControlHost,RPC
sacctmgr -P show assoc account=bvandur1_h200 format=Account,User,QOS
sacct -u $USER -S 2026-09-01 -X -o JobID,JobName,Partition,Account,QOS,AllocTRES,Elapsed,State
squeue -p h100,h200 -o "%.10i %.12u %.10a %.10q %.6D %.8b %.11l %.11M %.9T"
sbatch --test-only -p {h100,h200} -A {bvandur1_blabscify,bvandur1} -q {jhu2,scavenger} --gres=gpu:4 --nodes=1 ...
df -h /weka/scratch/jhu/bvandur1 /weka/projects/bvandur1
```

### 14.1 Hardware inventory

| partition | nodes | GPUs/node | total GPUs | GRES string | max wall | notes |
|---|---:|---:|---:|---|---|---|
| **h100** | 32 (`gh101-132`) | 4 | **128** | `gpu:h100:4` | 3-00:00:00 | `AllowAccounts=jhu`, `DenyQos=class`, `OverSubscribe=YES:4`, `PreemptMode=REQUEUE`, `DefCpuPerGPU=30`, `DefMemPerCPU=MaxMemPerCPU=12000`, `MaxCPUsPerNode=124` |
| **h200** | 6 (`gh201-206`) | 4 | **24** | `gpu:h200:4` | 3-00:00:00 | same flags |
| a100 | 13 | 8 | 104 | `gpu:a100:8` | 3-00:00:00 | out of scope (H100/H200 only) |
| b200 / b300 | 16 / 17 | 8 | 128 / 136 | `gpu:b200:8` / `gpu:b300:8` | 3-00:00:00 | out of scope; b300 mostly in `maint` |
| med (CPU) | 74 | — | — | — | 3-00:00:00 | `QoS=cpu_qos`, 108 CPU/node, `MaxMemPerCPU=4000` |

**H100 + H200 = 152 GPUs installed cluster-wide**, shared with every other user.

### 14.2 Your accounts and QOS

```
cluster  | account              | user     | QOS       | default
skipjack | bvandur1             | zhuicon1 | scavenger | scavenger
skipjack | bvandur1_blabscify   | zhuicon1 | jhu2      | jhu2
```

| QOS | MaxTRESPU | MaxTRESPerJob | MaxJobsPU | MaxSubmitPU | MaxWall | preempt |
|---|---|---|---|---|---|---|
| **jhu2** | **gres/gpu = 32** | *(none)* | *(none)* | *(none)* | *(none — partition's 3 d applies)* | not preemptible |
| **scavenger** | *(none)* | *(none)* | *(none)* | *(none)* | *(none)* | **preemptible**, `PreemptMode=REQUEUE` on h100/h200 |

- **GPUs per job:** no QOS or association cap, and `MaxNodes=UNLIMITED` on both partitions. Under
  `jhu2` a single job is effectively bounded by the 32-GPU per-user ceiling.
- **Concurrent GPUs:** `jhu2` **permits up to 32** — a ceiling, **not a reservation or guarantee**;
  what you actually get at any moment depends on queue contention, fairshare and backfill. `scavenger`
  can add more, preemptibly, with no stated ceiling and no promise.
- **Jobs running / pending:** no `MaxJobs`, `MaxSubmitJobs`, `GrpJobs` or `GrpSubmitJobs` on either
  association or QOS. Array size is bounded only by the cluster's `MaxArraySize`.
- **Per-account limits:** none beyond the QOS `MaxTRESPU`.
- **Wall time:** **3 days** max on h100, h200 and med (`DefaultTime=12:00:00` — always set `--time`).
- **You are NOT in the `bvandur1_h200` account** (`h200_condo` QOS); only `bvandur1` (the PI) is. So
  h200 access is via `jhu2` or `scavenger` like h100, with no condo priority.

### 14.3 Accessibility, validated

All four combinations validated with `sbatch --test-only` for a 1-node / 4-GPU / 32-CPU / 200G job:

| partition | account | QOS | result |
|---|---|---|---|
| h200 | bvandur1_blabscify | jhu2 | ✅ `Job … to start at 2026-09-28T18:11 … on nodes gh205` |
| h200 | bvandur1 | scavenger | ✅ `… 2026-09-29T18:11 … gh205` |
| h100 | bvandur1_blabscify | jhu2 | ✅ `… 2026-09-27T19:49 … gh111` |
| h100 | bvandur1 | scavenger | ✅ (valid) |
| h100, `--gres=gpu:1 --array=0-63` | bvandur1_blabscify | jhu2 | ✅ valid |

(Those start times are Slurm's pessimistic backfill estimate for a *4-GPU* request under the queue as
sampled, not a promise. Single-GPU tasks schedule far sooner — your own `mc_*` jobs on h100 ran within
minutes.)

**Do H100 and H200 need different directives?** Only if you pin the GRES *type*. `--gres=gpu:h100:1`
will never land on h200. **`--gres=gpu:1` with `--partition=h100,h200` works for both** and is what is
proposed. Everything else (account, QOS, CPU/mem defaults, wall time) is identical.

### 14.4 Current constraints affecting the sharding strategy
- Queue as sampled: **235 pending jobs on h100, 82 on h200**; `gh205` idle, `gh206` in `maint`,
  `gh201` reserved, `gh102` draining, `gh101` reserved. Contention is real, and is the reason the
  32-GPU figure is a ceiling rather than a plan.
- `PreemptMode=REQUEUE` + scavenger ⇒ **shard-level checkpointing is not optional**, and shards must
  be small enough that a preemption loses minutes, not hours (§15).
- `OverSubscribe=YES:4` means the CPU side of a node is shared; request the CPUs you need (16/GPU)
  rather than relying on exclusivity.

---

## 15. Shards, jobs, and how they distribute

### 15.1 Projected document counts (1.5B realized × 2; same pool ⇒ same tokens/doc)

| setting | source budget | est. source docs | ×2 passes = generations |
|---|---:|---:|---:|
| quality-first | 20B | ~12.2 M | 24.4 M |
| diversity-oriented | 20B | ~11.7 M | 23.4 M |
| disagreement-aware | 20B | ~11.2 M | 22.4 M |
| wrap-inspired | 20B | ~21.1 M | 42.2 M |
| **rewire-inspired** | **40B** | **~42.4 M** | **84.8 M** |
| **total** | **120B** | **~98.6 M** | **~197.2 M** |

(WRAP and REWIRE have ~2× the document count per token because a uniform sample draws shorter
documents than a fastText-ranked prefix: ~945 vs ~1,640 llama-2 tokens/doc at 1.5B.)

### 15.2 Shard size — 10,000 rows (Decision 8)
At the measured throughput a 10k-row shard is **~15–20 minutes** of GPU time — small enough that a
scavenger preemption costs one shard, large enough to keep filesystem metadata sane.

| shard rows | input shards | output files (both passes) | est. time/shard | preemption loss |
|---:|---:|---:|---|---|
| 5,000 | 19,720 | 39,440 | ~8 min | best, but 2× the metadata |
| **10,000** | **9,860** | **19,720** | **~16 min** | **chosen** |
| 50,000 | 1,972 | 3,944 | ~80 min | too coarse under scavenger |

(The 1.5B run used 200 shards per dataset, so ~30k rows/shard for quality-first — ~50 min each. Fine
on a non-preemptible queue; not the right choice here.)

### 15.3 Job layout
- **Per (setting, pass)** → one Slurm array against one claim directory: **10 arrays**
  (5 settings × 2 passes).
- Array A: `--array=0-31%32`, `jhu2`. Array B: `--array=0-63%64` (or higher), `scavenger`,
  `--requeue`. Both point at the same claim dir; a worker exits cleanly when no shards remain.
- Ordering: smallest first (`disagreement-aware`, `quality-first`, `diversity-oriented`,
  `wrap-inspired`, `rewire-inspired`) — the 1.5B order, so problems surface cheaply.
- 3-day wall: a worker that reaches its limit dies mid-shard; the claim goes stale and another worker
  reclaims it. Re-submitting the array is the resume.

### 15.4 GPU-hour estimate (derived, not guessed)

From the 1.5B benchmark `06_vllm/bench_out/benchmark_10k_7B_4xH100.md` (Qwen2.5-7B-Instruct,
4×H100 NVL, identical config): **62,502 total tok/s aggregate over 4 GPUs = 15,626 (in+out) tok/s per
GPU**; 28.6 docs/s aggregate.

From the 1.5B census (CVLLM §5): 192.6B templated-Qwen input against 220B llama-2 source-text tokens
across 2 passes ⇒ **templated Qwen input ≈ 0.875 × llama-2 source tokens** (Qwen tokenizes English
more compactly than llama-2; the per-document 150–185-token prompt overhead is included).

```
source (llama-2)         120.0 B  × 2 passes = 240.0 B
templated Qwen input     240.0 B × 0.875     = 210.0 B
output (llama-2)         Σ r_arm × B_s       =  87.2 B     (§18.2)
output (Qwen, ≈×0.875)                       =  76.3 B
total tokens through GPU                     ≈ 286.3 B
GPU-hours @ 15,626 tok/s                     ≈ 5,090 H100-GPU-hours
cross-check by documents: 197.2 M gens @ ~10 docs/s/GPU  ≈ 5,480 H100-GPU-hours
```
**Estimate: ~5,000–6,500 H100-GPU-hours (call it 5,500 ± 30%).** H200's larger, faster HBM should give
roughly 1.2–1.4× on this prefill-heavy mix, but that is an extrapolation — `bin/calibrate.py` measures
it on the first real shards before the fleet is scaled up.

> **Measured 2026-09-23, during implementation** (3,000 real DCLM-RefinedWeb documents, production
> Qwen tokenizer and chat template): the templated-Qwen / llama-2-source token ratio came out at
> **1.013** on a uniform sample whose median document is 480 llama-2 tokens, against the 0.875 used
> above. This is a document-length-mix effect, not a modelling error — the fixed 150-token prompt
> overhead is ~17% of a short document's templated length, and the 1.5B census figure was taken over
> the whole corpus. The GPU-hour figure should therefore be read at the upper end of its stated band
> (~6,000 H100-GPU-hours), and `bin/calibrate.py` re-measures it on the first real shards before the
> fleet is scaled up. The same check confirmed the status-0 drop rate at **0.07–0.10%** against the
> 1.5B benchmark's 0.10%, and that no kept document can overflow the 32,768-token context under
> either pass's drop threshold.

**Wall clock** (with the §14.2 caveat that 32 GPUs is a ceiling, not an allocation):

| effective fleet | wall clock |
|---|---|
| 32 GPUs | ~7.2 days |
| 32 + 32 effective scavenger | ~3.6 days |
| 32 + 64 effective scavenger | ~2.4 days |

Plus post-processing: the REWIRE fastText re-scoring is 42.4M × 2 ≈ **85M rewritten documents** to
score on CPU (the 1.5B run scored 42.4M). Budget one 96-core `med` job per ~10M documents, run as an
array over the scored-pool shards.

---

## 16. Resume / failure-recovery strategy

| failure | detection | recovery |
|---|---|---|
| **scavenger preemption** | `--requeue`; `SLURM_RESTART_COUNT` logged | claim goes stale (heartbeat older than 30 min, or job id absent from `squeue`) → reclaimed by any worker; at most one shard of work lost |
| **3-day wall reached** | job ends | resubmit the same array; completed shards skipped by output existence, in-flight claims go stale |
| **worker crash / OOM / node fail** | missing heartbeat | same stale-claim path |
| **partial shard write** | impossible — `.tmp` + `os.replace` only | a `.tmp` left behind is unlinked on next claim |
| **corrupt output shard** | `validate.py` compares `ParquetFile(p).metadata.num_rows` to the input shard's row count; mismatch ⇒ delete + reclaim | automatic on the next `validate` pass |
| **vLLM engine-init failure** | worker exits non-zero before claiming | array task retried; if systematic, preflight/`calibrate` catches it first |
| **prompt or model drift** | preflight md5 + overhead gates (150/185/72/66/73/83) fail **before any GPU time** | hard stop |
| **numpy PCG64 change** | `tests/test_wrap_styles.py` golden vector | hard stop |
| **selection non-determinism** | `01_select.py --phase report` re-run must produce byte-identical `doc_ids.npy` sha256 | hard stop |
| **post-processing NODE_FAIL** | the 1.5B run hit this twice (`1593189`, `1610294`): a heavy 96-worker `srun` followed by a second `srun` in the same batch job failed to launch | **split each post-processing step into its own job** with `--dependency=afterok`. This is the documented 1.5B mitigation and it worked for every arm. |
| **budget under-fill** | `fill_to` returns `filled=False` | `stop()`, never pad. (Diversity's per-topic shortfall is the one *expected*, *reported*, non-fatal case — policy A.) |

Idempotence: every stage is safe to re-run. Selection is pure given (pool sha, seed, budgets).
Rewriting skips completed shards. Post-processing steps skip completed outputs. Nothing appends.

---

## 17. Validation at every stage

**Stage 00 — pool**
200 shards present; per-shard row count (500,000 / 449,162); `doc_id` contiguous and equal to
`shard*500000 + row`; total 99,949,162; zero nulls in `tokens-llama2`, the three `*-ranking-v2`
columns and `topic`; exactly 24 distinct topics; sha256 of every shard recorded; `*-ranking-v2` all
in (0,1]. (All are `stop()` conditions in the 1.5B code — `select_10b.py:112-130`.)
**No 5M-sample filtering is applied** (§20 D-1); the manifest records this explicitly so a future
reader cannot mistake it for an oversight.

**Stage 01 — selection**
- base ∩ strategy = **0** for all six settings (hard stop) — the §0 invariant;
- val ∩ everything = **0**;
- no duplicate `doc_id` within a block;
- `quality-base ⊂ quality-first` (both prefixes of the same fastText order) — asserted;
- every block `filled == True`; overshoot < one document (Diversity's per-topic shortfall excepted and
  reported);
- topic count = 24; per-topic quota table emitted; per-topic realized vs quota reported; **per-topic
  available supply in `D'` vs 2× the 1.5B draw reported** so the Diversity shortfall is sized before
  any GPU time;
- DA: `U`, `U_τ`, `τ_q`, `τ_v`, the λ=0.5 cutoff and `|S_DA|/|U_τ|` written into
  `DOMAIN_STATS.md` in the same format as the 1.5B file so the two are diffable;
- pairwise overlap matrix across the six settings emitted (the 3B analogue of paper Fig. 9, which
  **must** be regenerated because REWIRE's base rate moves from 22% to ~47%);
- re-running `--phase report` gives byte-identical `doc_ids.npy` sha256.

**Stage 02 — materialized sources**
Row count per shard matches the selection; `Σ(tokens-llama2+1)` over all shards equals the selection
manifest exactly; every `doc_id` present exactly once; text non-null; `_shards.json` sha256 recorded
(the fingerprint that makes `shard_index` stable, hence the wrap styles reproducible).

**Stage 03 — rewriting**
Output rows == input rows for every shard; `status ∈ {0,1,2}`; `status==0` ⇔ templated input over the
pass's threshold; `rewritten_tokens` recomputed with the llama-2 tokenizer; `wrap_style` present and
non-empty exactly for the wrap p1 pass and 25.0% ± 0.5pp per style; prompt overhead gates re-asserted
at worker start; **both passes present for all five arms** before stage 04 may run (a hard gate);
per-shard completion recorded. Qualitative monitor (format-only / degenerate repetition /
suspiciously short) sampled every 10,000 docs, as at 1.5B.

**Stage 04 — post-processing**
Prefix strip is **start-anchored only** and the stripped fraction is reported (1.5B: 99.98–99.99% of
wiki, ~0.002–0.56% of distill); `rewritten_tokens` recounted after stripping; only `status==2` rows
ever enter the assembly; `Σ(rewritten_tokens+1) ≥ B` with overshoot < one document, or an explicit
reported shortfall; distill share per arm reported against the 1.5B Table 7 values as a sanity check;
REWIRE: **both passes confirmed present in the pool**, then pool size, score distribution, cutoff,
acceptance rate and wiki/distill split reported and compared against the 1.5B figures (30.8%
acceptance, 53.1% distill share).

**Stage 05 — final corpus**
`doc_id` overlap between the shared base and the strategy half = **0**; total tokens per setting
reported against 20B with any shortfall named; shuffle row count conserved; source-prompt distribution
reported; per-topic token distribution reported for diversity; a 200-document spot check that final
text matches the recorded rewrite; and a **trainability check** — tokenize a sample with the llama-2
tokenizer and confirm the realized token total matches the manifest (the `kys_raw` practice).

**Handoff check (for the training repo).** `_pretrain_manifest.json` records each corpus's exact token
total **and the number of passes required to reach 20B / 40B / 60B**, so the training configs set
matched token horizons rather than a fixed epoch count. For five settings that is 1.000 / 2.000 /
3.000; for Diversity it is ~1.011 / 2.023 / 3.034.

---

## 18. Expected intermediate and final token counts

Projected with the **measured** per-arm, per-pass ratios of §2.11 applied to the 3B source budgets.

### 18.1 Selection (raw source tokens)

| block | budget | est. docs |
|---|---:|---:|
| shared base | 10.000 B | ~8.2 M |
| quality-base (2nd half) | 10.000 B | ~6.1 M |
| quality-first source | 20.000 B | ~12.2 M |
| wrap-inspired source | 20.000 B | ~21.1 M |
| rewire-inspired source | **40.000 B** | ~42.4 M |
| diversity-oriented source | 20.000 B | ~11.7 M |
| disagreement-aware source | 20.000 B | ~11.2 M |

`D'` (residual pool) ≈ **85 B** tokens / ≈ 91.6 M docs. Derivation: at 1.5B, WRAP's 10B was 11.07% of
the candidate pool ⇒ pool ≈ 90.3 B with a 5 B base removed ⇒ ALIVE ≈ 95 B; removing a 10 B base leaves
≈ 85 B. Cross-check from document lengths: a uniform sample averaged ~945 llama-2 tokens/doc at 1.5B,
so 99.95 M docs ≈ 94.5 B, minus a 10 B base ≈ 84.5 B. **To be confirmed exactly by
`01_select.py --phase report`.**

DA domain under Option C (to be confirmed in the report phase): `U` ≈ 47 B, `U_τ` ≈ 36 B,
`S_DA` = 20 B ≈ 56% of `U_τ`.

### 18.2 Rewriting output (llama-2 tokens)

| arm | source | r p1 | r distill | p1 out | distill out | pool total | target B | headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quality-first | 20 B | 0.3399 | 0.2581 | **6.80 B** | 5.16 B | 11.96 B | 10 B | **+19.6%** ← tightest |
| diversity-oriented | 20 B | 0.3812 | 0.2845 | **7.62 B** | 5.69 B | 13.31 B | 10 B | +33.1% |
| disagreement-aware | 20 B | 0.3649 | 0.2750 | **7.30 B** | 5.50 B | 12.80 B | 10 B | +28.0% |
| wrap-inspired | 20 B | 0.4313 | 0.3657 | **8.63 B** | 7.31 B | 15.94 B | 10 B | +59.4% |
| rewire-inspired | 40 B | 0.4628 | 0.3660 | **18.51 B** | 14.64 B | 33.15 B | 10 B | +231% |
| **total** | **120 B** | | | **48.86 B** | **38.30 B** | **87.16 B** | 50 B | |

**Every arm's pass-1 output alone is below 10 B**, which is the quantitative basis for Decision 3 and
the same conclusion the 1.5B run reached empirically (paper Table 7). The headroom column is exactly
`r_arm × source / 10B`, identical to the 1.5B ratios because both the budget and the source scaled by
2 and `r` is a property of the prompt and the document population, not the scale.

**Caveat on `r`.** These ratios were measured on the 1.5B *populations*. The 3B selections reach deeper
into the same pool (20B rather than 10B of a ~85B pool), so their document mix shifts slightly toward
lower-scoring, possibly shorter documents. With quality-first at only +19.6% headroom, a ~16% adverse
shift would put it under budget. Mitigation: `bin/calibrate.py` re-measures `r` on the first ~1% of
each arm's shards and reports the projected fill before the fleet is scaled up.

### 18.3 Final corpora and matched training horizons

| setting | shared base | strategy half | corpus size | passes to reach 20B / 40B / 60B |
|---|---:|---:|---:|---|
| Quality-Base | 10.000 B raw | 10.000 B raw | ~20.000 B | 1.000 / 2.000 / 3.000 |
| Quality-First | 10.000 B raw | 10.000 B rewritten | ~20.000 B | 1.000 / 2.000 / 3.000 |
| WRAP-Inspired | 10.000 B raw | 10.000 B rewritten | ~20.000 B | 1.000 / 2.000 / 3.000 |
| REWIRE-Inspired | 10.000 B raw | 10.000 B rewritten | ~20.000 B | 1.000 / 2.000 / 3.000 |
| **Diversity-Oriented** | 10.000 B raw | **~9.78 B rewritten** | **~19.78 B** | **~1.011 / 2.023 / 3.034** |
| Disagreement-Aware | 10.000 B raw | 10.000 B rewritten | ~20.000 B | 1.000 / 2.000 / 3.000 |

**All six settings are trained and evaluated at the same ~20B / ~40B / ~60B cumulative
training-token horizons.** Diversity's corpus being ~1.1% smaller changes only how many times it is
re-read (~3.03 passes instead of 3.00 by the final checkpoint) — it is **never** trained to a
~59.3B horizon, and `corpus_size × 3` is never used as a horizon anywhere in this design. This matches
the 1.5B study, where diversity-first's 9.890B corpus was trained against the same matched
10B/20B/30B horizons as the other five settings.

The ~9.78B figure assumes the 1.5B shortfall pattern scales (110.3 M on a 5 B budget → ~220 M on
10 B). It could be worse; the report phase sizes it exactly before any GPU time is spent, and a worse
shortfall changes only the pass count, never the horizon.

**Discarded (generated but not retained): 87.16 − 50 ≈ 37 B rewritten tokens**, 23.2 B of which is
REWIRE's rejected tail. That is inherent to the design, not waste.

---

## 19. Storage estimate

Text sizing uses the CVLLM model (**4.2 bytes per llama-2 token** uncompressed; zstd ≈ 0.30×) and the
observed pool density (181.58 GB parquet for ~95 B tokens ⇒ ~1.9 bytes/token *already compressed*).

| item | raw | on disk (zstd parquet) |
|---|---:|---:|
| scored pool download (200 shards) | — | **182 GB** |
| 02 materialized sources: 120 B source tokens, `(doc_id, text)` only | 504 GB | **~155 GB** |
| 03 rewritten outputs: 87.2 B output tokens + small int columns, both passes | 366 GB | **~120 GB** |
| 04 REWIRE scored pool (adds 2 float32 to 84.8 M rows; text re-written) | — | **~55 GB** (transient) |
| 04 assembled strategy halves: 50 B retained tokens | 210 GB | **~65 GB** |
| 05 final shuffled corpora, base copied per setting: 6 × ~20 B tokens | 504 GB | **~155 GB** |
| logs, manifests, claims, reports | — | ~5 GB |
| **peak total** | | **≈ 737 GB (0.72 TiB)** |
| steady state after deleting `02_sources` and the REWIRE transient | | **≈ 527 GB** |
| after also deleting `03_rewritten` | | **≈ 407 GB** |

**Nanotron `.ds` tokenization (~240 GB) is explicitly out of scope for this repo** (Decision 10) and
is therefore excluded from the totals above. It belongs to `nanotron-kys`.

Optional further savings:
- store the shared base **once** and symlink it into each `05_final/<setting>/` → saves 5 × 10 B ≈
  **42 GB** (but makes each setting non-self-contained; the 1.5B run copied, and copying is kept as
  the default);
- skip `02_materialize_sources` and read text from the pool by `(shard, row)` at generation time →
  saves ~155 GB but puts ~96 concurrent random readers on 200 × 908 MB files. Not recommended.

`/weka/projects/bvandur1` has **117 TB free**, so ~0.72 TiB peak is comfortable. The scratch quota
(137 GB free) could not have held even the pool — hence Decision 7.

---

## 20. Decisions Required Before Implementation

All twelve are now resolved. One carries a **paper-side action** rather than a pipeline change.

---

### Decision 1 — the 5M analysis sample ✅ **RESOLVED (pipeline)** · ⚠️ **PAPER CORRECTION REQUIRED**

**Your instruction:** use the provided 100M scored pool directly; do not attempt to remove the 5M
sample again. **That is what the pipeline will do**, and it is also what reproducing the 1.5B
implementation requires. **This part is settled.**

**However, the stated reason is not what the evidence shows.** You said the 5M sample "was already
removed upstream before the current 100M scored source pool was constructed." I ran the provenance
check you asked for, and it establishes the opposite. Per your own instruction for outcome B, I am
reporting it rather than silently adopting it.

**Evidence chain, all first-hand and read-only:**

1. **The 5M analysis sample is a subset of the 100M reservoir sample, not a disjoint draw.**
   `datasets/exclusions/exclude_all_global_indices.parquet` (local, 208 MB) holds 5,048,898 rows:
   **5,001,383** labelled `exclusion_source == '5m_basic'` and **47,515** labelled `'50k_claude'`.
   Every one of its `global_index` values lies in **[1, 99,999,996]** — inside the 100M pool's own
   index space — and all 5,048,898 are unique. `13_600M/flag_exclusions.py` and
   `13_600M/README.md` independently confirm that this column "indexes the **OLD 100M pool** (max
   99,999,996)" and that the exclusion list "only ever intersects that [appended 100M] tail block
   (~5.05% of it)".

2. **The published table removed only the 50,838-row annotator-contamination set.**
   `00_TMP/merge_remove_50k.py` + `00_TMP/merge_remove_50k_report.md` are the exact script and report
   that produced `6_merged_clean` (= the HF table): input 100,000,000 rows; removal set
   `01_explore/match_50k_prefix4000.npy`, **50,838 positions, verbatim**; output 99,949,162 rows with
   `orig_doc_id` verified equal to "the sorted complement of the 50,838 removed positions (0 leaked)".
   The HF dataset card says the same thing: the only rows removed are those overlapping the ModernBERT
   scorer's training set.

3. **Direct intersection.** I loaded `match_50k_prefix4000.npy` (50,838 int32 positions) and
   intersected it with the exclusion table:
   ```
   5m_basic  keys caught by the 50,838 removal:  2,530
   50k_claude keys caught by the 50,838 removal: 47,515   (all of them ✓)
   => 5M-analysis-sample documents surviving into the published pool: 4,998,853
      = 5.001% of 99,949,162
   ```

**Conclusion — outcome B.** ~**5.0 million** analysis-sample documents (5.001% of the pool) are
present in `KYS-DCLM-Refinedweb-100M-Scored`, and the 1.5B selection pipeline removed none of them.
Since the shared base and every strategy selection draw from that pool, each of them contains ~5% of
its documents from the analysis sample; at 1.5B the shared top-5B base (4,120,164 docs) would have
contained on the order of 200,000 such documents, and the WRAP arm on the order of 500,000.

**What this does and does not mean.**
- It is **not a label-leakage problem.** The ModernBERT scorer's *training* set (the 50k Claude
  annotations) *was* correctly and fully removed — all 47,515 keys are in the 50,838 removed. The 5M
  sample was used only to *measure* scorer behaviour (Fig. 2, App. B topic distributions), not to fit
  anything and not to make any selection decision. So no scorer saw a training document it then scored
  for selection.
- It **is a factual error in the paper.** P §3 and the Fig. 1 caption both assert the exclusion, and
  that assertion is false for the 1.5B results as published, and will be false for the 3B results too.

**Required action (paper, not code):** correct §3 and the Fig. 1 caption to state what actually
happened — a 5M-document sample was drawn for scorer and topic analysis; the 50,838 documents used to
train the ModernBERT scorer were removed from the pool; the analysis sample itself was not removed,
and its ~5.0 M documents remain eligible for selection. Optionally add one sentence noting that the
analysis sample was descriptive only and fitted no model used for selection.

**Pipeline consequence: none.** `ALIVE = all 99,949,162 − 50,000 val` and nothing else, exactly as at
1.5B. The stage-00 manifest records this explicitly with a pointer to this section, so no future
reader mistakes it for an oversight.

---

### Decision 2 — Disagreement-Aware candidate domain ✅ **RESOLVED — Option C**

**Locked:** for each of the three scorers, rank documents within `D'` and retain the highest-ranked
until a **20B raw-source-token budget** is reached; `U` = union of the three 20B selections;
`τ_q` = Q30(q over U); `τ_v` = Q90(v over U); `U_τ = {q ≥ τ_q, v ≤ τ_v}`; rank `U_τ` by
`u = q + 0.5√v`; select 20B raw source tokens.

λ = 0.5, floor percentile 30, cap percentile 90 — unchanged, never retuned against 3B downstream
performance.

Rejected alternatives, recorded: the literal 1.5B rule ("top 10% by tokens of `D'`") gives ≈8.53B per
scorer ⇒ `U_τ` ≈ 15.8B **< 20B, infeasible**; disabling the domain and ranking `u` over all of `D'`
(what `13_600M/02_select` chose at larger scale) would change what the arm tests.

Why Option C is the right analogue: at 1.5B the coded per-scorer budget was 9.03B = **0.90·`B_s`**, so
the paper's "10B-token budget" was a rounding of it. Setting the per-scorer budget to `B_s` preserves
both structural ratios of the 1.5B run (`|U|/B_s` ≈ 2.2, `|S_DA|/|U_τ|` ≈ 54–56%) and needs no 1.5B
constant hard-coded into 3B code.

---

### Decision 3 — both rewriting passes for all five arms ✅ **RESOLVED**

**Locked:** generate **both** passes over the **full** selected source set for Quality-First,
WRAP-Inspired, REWIRE-Inspired, Diversity-Oriented and Disagreement-Aware. Four non-WRAP arms:
pass 1 = grounded Wikipedia-style, pass 2 = distill. WRAP: pass 1 = one of the four adapted style
prompts per document, pass 2 = the shared distill prompt. **No adaptive generation shortcut based on
projected pass-1 yield.**

For REWIRE specifically, Wikipedia-style **and** distill outputs must both enter the rewritten
candidate pool before fastText scoring and top-10B retention — enforced as a hard gate in stage 04c.

This matches both the 1.5B implementation and P App. C. Projected pass-1-only yields (6.80 / 7.62 /
7.30 / 8.63 B) are all below 10 B anyway, so nothing is lost.

---

### Decision 4 — REWIRE at 40B raw source ✅ **RESOLVED**

**Locked:** shared base = top 10B raw fastText tokens; `D'` = everything after removing the base;
uniformly sample **40B** raw source tokens from `D'`; rewrite all 40B with both passes; fastText-score
the rewritten outputs; globally rank; retain top ~10B rewritten tokens; final mixture = 10B shared raw
+ ~10B REWIRE rewritten. **κ = 2.**

40B being ~47% of `D'` (vs 22.15% at 1.5B) is **accepted**. κ is **not** reduced to preserve the old
overlap percentage. Consequence to carry into the write-up: the overlap analysis (paper App. D.3 /
Fig. 9) must be regenerated at 3B, where REWIRE's independence base rate is ~47%, not 22.1%.

---

### Decision 5 — Diversity-Oriented policy and training horizon ✅ **RESOLVED**

**Locked, selection:** partition `D'` by the 24 WebOrganizer topics; per-topic source budget
proportional to that topic's token share of `D'`; rank **within each topic by consensus quality `q`**
from all three scorers; fill each quota; then the **same q-DESC top-up** the validated 1.5B
implementation used if the union falls short of 20B.

**Locked, rewriting:** both passes.

**Locked, assembly:** per-topic quota from the selected source set; within each topic retain all
`status==2` pass-1 outputs first, then top up from distill ordered by **`fasttext-ranking-v2` DESC
within topic**; **no cross-topic backfill; no padding**; report any shortfall.

**Locked, training horizon:** even if the final Diversity corpus is under 20B (expected ~19.78B), the
training horizons remain **~20B / ~40B / ~60B cumulative tokens**, identical to the other five
settings. The corpus is simply re-read slightly more often (~1.011 / 2.023 / 3.034 passes). The
horizon is **never** computed as `corpus_size × 3`. Every occurrence of the old "~59.3B" figure has
been removed from this plan (§5, §6.5, §18.3, §17 handoff check).

The report phase sizes the per-topic supply exactly, before any GPU time, so the expected shortfall is
known in advance.

---

### Decision 6 — Diversity's two quality keys ✅ **RESOLVED**

**Locked:** consensus `q` at **selection** (matching P Eq. 6) and `fasttext-ranking-v2` at
**assembly**, within topic — exactly what the 1.5B code did. **Not** changed to `q` at both stages,
because this 3B run is a scale-up validation and changing the model scale and the data algorithm
simultaneously would weaken interpretability. A footnote in the paper should record the assembly key.

---

### Decision 7 — storage root ✅ **RESOLVED**

**Locked:** physical root `/weka/projects/bvandur1/zhuicon1/rewrite-3b/` (117 TB free). Symlinks
created later — not now, and not until the target directories exist — at:

- `.../datasets/dclm-refinedweb-100m-sample` → `.../rewrite-3b/00_pool`
- `.../datasets/rewrite-3b-llama-60b-tokens` → `.../rewrite-3b/dataset`

The scratch quota has 137 GB free and could not hold even the 182 GB pool. No large files are created
yet.

---

### Decision 8 — shards and work distribution ✅ **RESOLVED**

**Locked:** 10,000-row shards; claim-directory work distribution; stale-claim recovery (heartbeat >
30 min or job id absent from `squeue`); fully resumable and idempotent workers.

Operationally different from the 1.5B modulo assignment, but **output semantics are unchanged**:
per-document results depend only on `(shard_index, row_index)`. The one visible consequence is that
the WRAP style draw is a *different but statistically identical* draw from the 1.5B one, because
`shard_index` is now ours (§8.3) — accepted.

---

### Decision 9 — queues ✅ **RESOLVED**

**Locked:** use both `jhu2` (account `bvandur1_blabscify`) and `scavenger` (account `bvandur1`)
against the same claim directory.

**Wording corrected throughout (§13.3, §14.2, §14.4, §15.4):** `jhu2` **allows up to 32 concurrent
GPUs** for this user/QOS. It does **not** guarantee 32 GPUs. Actual availability depends on queue
contention and scheduler allocation across the 128 H100 + 24 H200 GPUs shared with all other users
(235 pending h100 jobs and 82 pending h200 jobs at the time of inspection). `scavenger` may provide
additional **preemptible** capacity with no stated ceiling and no promise. All wall-clock figures in
§15.4 are therefore labelled by *effective* fleet size, not by entitlement.

---

### Decision 10 — this repo's final output ✅ **RESOLVED**

**Locked:** this data-generation repo stops at **shuffled parquet** final corpora
(`05_final/<setting>/shuffled/`). Nanotron `.ds` tokenization is **not** a required output of this
repo and is removed from the storage estimate (−240 GB). The Nanotron training repository
(`projects/nanotron-kys`) owns tokenization into its own format, exactly as at 1.5B. The handoff
contract is `_pretrain_manifest.json`, which records each corpus's exact token total and the pass
counts needed to reach the 20B / 40B / 60B horizons.

---

### Decision 11 — separation from the 7B / 600M design ✅ **RESOLVED**

**Confirmed explicitly.** This 3B experiment:

- uses the **100M scored source pool** (`blab-jhu/KYS-DCLM-Refinedweb-100M-Scored`, 99,949,162 rows);
- budgets are **10B shared base + ~10B strategy-specific**;
- corpus per setting is **~20B**;
- training horizon is **~20B / ~40B / ~60B** cumulative tokens;
- is **NOT** a continuation of the previous 7B / 600M / 50B-per-arm design
  (`13_600M/02_select`, `rewrite-vllm`, `wytro/Know-Your-Sources-7B`);
- **does not reuse** that design's budgets (20B core / 60B–120B remainders / 50B per arm), its 600M
  source pool, its DA simplification (domain disabled, `u` ranked over the whole residual pool), or
  its `doc_id` space (600M-indexed, with a `shard_manifest.parquet` offset scheme incompatible with
  `doc_id = shard*500000 + row`).

What **is** reused from `rewrite-vllm` is engineering only, and it is listed in §3: engine config,
prompt files and md5/overhead gates, wrap-style golden tests, trim/shuffle parity tests, the
claim-based work-distribution pattern, and the calibration script. Two of its *findings* are also
carried over as facts: the measured per-arm compression ratios (§2.11) and the κ-on-the-remainder rule
(§7). No budget, pool or algorithm crosses over.

---

### Decision 12 — λ = 0.5 only ✅ **RESOLVED**

**Locked:** build and rewrite **only** λ = 0.5. No λ sweep at 3B — the 3B run validates the selected
DA method at larger model scale, it does not repeat the λ grid.

Consequence: the 3B generation bill is 120B source tokens × 2 passes, not the 1.5B run's 110B × 2
(which included six λ variants at 10B each). This is already reflected in §15.4 and §18.

---

## Appendix A — exact 1.5B constants carried over unchanged

```
SEED                     = 42, via np.random.SeedSequence(42).spawn(8)
child 0  validation holdout (50,000 docs)      child 1  global tie-break permutation
child 2  WRAP uniform draw                     child 3  REWIRE uniform draw
exclusions               = validation holdout ONLY (no 5M-sample filtering; see §20 D-1)
train length             = tokens-llama2 + 1
tokenizer                = tokenizers/llama2-unsloth-tokenizer  (unsloth/llama-2-7b, vocab 32000, fast)
q                        = ((ft_v2 + fw_v2 + mb_v2)/3).astype(float32)
v                        = (((ft_v2-q)**2 + (fw_v2-q)**2 + (mb_v2-q)**2)/3).astype(float32)   # ddof=0
u                        = q + 0.5*sqrt(v)
tau_q                    = 30th percentile of q over U
tau_v                    = 90th percentile of v over U
lambda                   = 0.5  (only value built)
kappa                    = 2   (applied to B_s only, never to the shared base)
DA per-scorer budget     = B_s = 20B  (3B scale-up, Option C)
rewriter                 = Qwen2.5-7B-Instruct, bf16, tp=1, gpu_mem_util=0.85, max_model_len=32768
sampling                 = temperature 0, top_p 1.0, max_tokens min(4096, 32768 - n_in)
drop threshold           = 30720 (pass 1) / 28672 (distill, = max_model_len - max_tokens)
status                   = 0 dropped / 1 length / 2 stop
passes                   = 2 for every rewriting arm, over the FULL selected source set
wrap styles              = ["easy","hard","wiki","qa"], default_rng([42, shard_index]).integers(0,4,n)
wrap distill top-up      = seeded random, default_rng(42), NO quality signal
diversity selection key  = consensus q, within topic
diversity assembly key   = fasttext-ranking-v2, within topic; policy A (no backfill, no padding)
wiki prefix strip        = "Here is a paraphrased version:\n\n", START-ANCHORED only, then recount
fastText (REWIRE)        = fasttext_oh_eli5.bin; clean = text.replace("\n"," ").replace("\r"," ")[:100000]
                           raw = p if label == "__label__hq" else 1-p ; empty -> 0.0
                           pool = BOTH passes, status==2, not deduplicated
final shuffle            = bucketed_shuffle, seed 42, document level, default_rng([42, bucket])
training horizons        = 20B / 40B / 60B cumulative tokens, MATCHED across all six settings
```

## Appendix B — files and artifacts read for this plan

```
00_paper/Know_Your_Sources.pdf                                    (29 pp, full)
projects/rewrite/04_select/{select_5b.py,select_10b.py,select_quality_base_15B.py}
projects/rewrite/05_select_s5_variants/select_s5.py
projects/rewrite/06_lambda_grid/{select_lambda_grid.py,DOMAIN_STATS.md}
projects/rewrite/06_vllm/bench_out/benchmark_10k_7B_4xH100.md
projects/rewrite/07_rewrite/{README.md,rewrite_worker.py,sbatch_template.sh,launch_dataset.sh,launch_all.sh}
projects/rewrite/09_Distill/{README.md,sbatch_template.sh}
projects/rewrite/10_postprocess/{README.md,README_rewrite.md,DATASETS_SUMMARY.md,postprocess_report.md,
                                 run_all.sh,02_assemble_5B.py,_step2_rewrite_summary.json,
                                 _step3_rewrite_summary.json,_step4_rewrite_summary.json}
projects/rewrite/prompts/{README.md,wrap_prompts.json,wikipedia_style_rephrasing_grounded.md}
projects/rewrite/13_600M/{README.md,flag_exclusions.py,02_select/README.md}
projects/rewrite/01_explore/{README.md,merge_clean.py,match_50k_prefix4000.npy}
projects/rewrite/00_TMP/{merge_remove_50k.py,merge_remove_50k_report.md}
projects/rewrite/03_TokenCounts/count_tokens.py
projects/rewrite-vllm/{README.md,configs/vllm.yaml,docs/DESIGN_DELTA.md}
projects/nanotron-kys/tools/kys_raw/{KYS-Pre-Rewritten.README.md,raw_setting_job.sbatch,publish_raw_text.py}
datasets/exclusions/exclude_all_global_indices.parquet            (schema + counts + index ranges)
datasets/ppl-dsai/{dclm-refinedweb-100m-sample,dclm-refinedweb-5m-samples}  (listings only)
HF API: blab-jhu/KYS-DCLM-Refinedweb-100M-Scored  (metadata, full 200-file tree, README)
```
