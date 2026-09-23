# rewrite-3B / 01_data

Data selection, rewriting and post-processing for the **3B scale-up** of
*Know Your Sources: Data Selection Matters when Rewriting for Data-Constrained Pretraining*.

The behavioural source of truth is the validated 1.5B implementation in
`projects/rewrite/`.  Every design decision, every deviation, and the evidence behind each is
recorded in **[`plan.md`](plan.md)** -- read section 0 (locked decisions) first.

**This repository stops at validated shuffled parquet corpora.**  Tokenization into Nanotron's
format is the training repository's job.

---

## The six settings

Each corpus is `10B shared raw base + ~10B strategy-specific`, and every setting is trained to
the **same** cumulative token horizons: **20B / 40B / 60B**.

| setting | strategy source budget | rewrite | strategy half built by |
|---|---:|---|---|
| Quality-Base | 10B | no | next-10B fastText prefix, raw |
| Quality-First | 20B | wiki + distill | all pass-1, then fastText-ordered distill top-up |
| WRAP-Inspired | 20B | 4 styles + distill | all pass-1, then **seeded-random** distill top-up |
| REWIRE-Inspired | **40B** (κ=2) | wiki + distill | **fastText on the rewritten output**, global top-10B |
| Diversity-Oriented | 20B | wiki + distill | per-topic quota, fastText DESC within topic, policy A |
| Disagreement-Aware | 20B | wiki + distill | all pass-1, then `u = q + 0.5√v` ordered distill top-up |

Budgets are **TRAIN tokens** = `tokens-llama2 + 1` (one leading BOS), the 1.5B convention.
`λ = 0.5` only.  Every strategy draws **only** from the residual pool `D'` left after the shared
base is removed; `|BASE ∩ S| == 0` is a hard stop for all six.

---

## Layout

```
configs/      budgets.yaml (every budget + its 1.5B ancestor) | vllm.yaml | paths.yaml | cluster.yaml
prompts/      byte-copies of the six 1.5B prompts + PROVENANCE.md (md5 + overhead table)
src/kys3b/    the library: pool, select, disagreement, shards, claims, rewrite/, post/
bin/          one entry point per stage, plus preflight / validate / status / calibrate
slurm/        one sbatch per stage; the rewriting array is driven by bin/03_rewrite_launch.py
tests/        stdlib unittest; includes differential parity against the 1.5B source
scripts/      00_setup_env.sh (the vLLM env) | 01_make_symlinks.sh
```

Large data lives on `/weka/projects/bvandur1/zhuicon1/rewrite-3b/` (117 TB free); the scratch
quota has ~137 GB free and cannot hold the 182 GB pool.  `scripts/01_make_symlinks.sh` keeps the
originally specified scratch paths resolving.

---

## Running it

Everything except the GPU worker runs under `envs/data`:

```bash
E=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
export PYTHONPATH=$PWD/src
```

### 0. Gates and tests (no GPU, no download, no jobs)

```bash
$E tests/run_all.py -v          # the full suite
$E bin/preflight.py             # prompts, tokenizers, fastText, budgets, storage
```

### 1. Pool, then the report phase

```bash
sbatch slurm/download.sbatch                  # 182 GB, gated repo, needs your HF token
$E bin/validate.py --stage pool
sbatch slurm/select.sbatch report             # writes ONLY reports/ -- nothing under 01_selection/
```

The report phase answers, before a single GPU-hour is spent: the exact size of `D'`, the
Option-C DISAGREEMENT-AWARE domain (`U`, `U_τ`, `τ_q`, `τ_v`, cutoff), the per-topic
DIVERSITY supply against its quota (hence the expected shortfall), and the pairwise overlap
matrix -- which **must** replace the 1.5B figure, because REWIRE's independence base rate moves
from ~22% to ~47% when its pool goes from 20B to 40B.

### 2. Commit the selection and materialize the rewriting input

```bash
sbatch slurm/select.sbatch commit
$E bin/validate.py --stage selection
sbatch slurm/materialize.sbatch
$E bin/validate.py --stage sources
```

### 3. Rewriting -- both passes, all five arms

```bash
bash scripts/00_setup_env.sh                  # one-off: the vLLM env
$E bin/03_rewrite_launch.py                   # PRINTS the 10 sbatch lines, submits nothing
$E bin/03_rewrite_launch.py --test-only       # validate them (creates no job)
$E bin/03_rewrite_launch.py --queue both --submit     # actually queue
$E bin/status.py                              # progress + claim state
$E bin/calibrate.py                           # measured r vs the 1.5B census
$E bin/validate.py --stage rewritten
```

Both passes are **always** generated over the full source set (Decision 3).  For REWIRE this is
not an optimization question: the fastText filter ranks pass-1 and distill *together*, and
distill won 53.1% of the retained tokens at 1.5B.

### 4. Post-process, then assemble

One step per job -- the 1.5B run hit NODE_FAIL twice when two heavy `srun`s shared a batch job.

```bash
S=$(sbatch --parsable slurm/post.sbatch strip)
C=$(sbatch --parsable --dependency=afterok:$S slurm/post.sbatch score-rewire)
A=$(sbatch --parsable --dependency=afterok:$C slurm/post.sbatch assemble)
sbatch --dependency=afterok:$A slurm/final.sbatch
$E bin/validate.py --stage filtered
$E bin/validate.py --stage final
```

---

## What the design guarantees

* **Determinism.** `SeedSequence(42).spawn(8)` with a documented child map; `fill_to` keeps the
  last document whole; `order_desc` breaks ties with the child-1 permutation (load-bearing --
  the DCLM fastText percentile has an 8.6M-document tie at its floor).  Re-running the report
  phase must reproduce byte-identical `doc_ids.npy` digests.
* **`stop()`, never `warn()`.** Under-filled budget, wrong topic count, non-contiguous `doc_id`,
  base∩strategy overlap, prompt md5 or overhead mismatch -- all abort non-zero.  The one
  reported, non-fatal exception is the DIVERSITY per-topic shortfall (policy A).
* **Atomic writes only**, `.tmp` + `os.replace`, partial temps unlinked in `finally`.
* **Resumability.** Completion is "the output shard exists with the right row count", so a rerun
  is idempotent.  Shards are claimed through a directory with stale-claim recovery, so a
  preempted scavenger task costs one 10,000-row shard, not a whole array index's backlog.
* **Manifests are the contract.** Each stage validates the previous stage's manifest and records
  the code commit.

## Decision 1, in one paragraph

The published 99,949,162-row pool is used **exactly as it is**; nothing is filtered out except
the 50,000-document validation holdout.  The 5M scorer/topic analysis sample was **not** removed
upstream: 4,998,853 of its documents (5.001% of the pool) are present, and the 1.5B pipeline did
not remove them either.  The pipeline therefore reproduces 1.5B faithfully; it is the
manuscript's §3 / Fig. 1 claim that needs correcting.  Evidence and the exact provenance chain
are in `plan.md` section 20 D-1, and the stage-00 manifest records this so no future reader
mistakes it for an oversight.
