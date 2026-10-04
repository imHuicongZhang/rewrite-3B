#!/usr/bin/env bash
#SBATCH --job-name=kys3b_rewrite
#SBATCH --partition=h100,h200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --open-mode=append
#
# One array task = one GPU = one vLLM process at a time (tensor_parallel_size=1), exactly as at 1.5B.
# `--gres=gpu:1` uses the GENERIC type so a single array can land on either h100 or h200;
# `--gres=gpu:h100:1` would never schedule on h200.
#
# account / qos / array / --requeue are supplied by bin/03_rewrite_launch.py:
#   primary       : -A bvandur1_blabscify -q jhu2      (QOS MaxTRESPerUser = MaxTRESPerAccount = 32 GPUs)
#   opportunistic : -A bvandur1           -q scavenger --requeue (preemptible; no QOS ceiling)
# Every task of both arrays runs this same script against the SAME claim directories, so they
# cooperate.  A worker exits cleanly when no shard of its (setting, pass) is claimable.
#
# KYS_WORKLIST is a '+'-separated list of setting:pass items, processed in order, one worker
# process per item.  A task therefore moves on to the next (setting, pass) when one drains instead
# of exiting, which keeps every allocated GPU busy without one pending array per pair.  The list
# is walked TWICE: the second sweep picks up shards whose owner died after this task passed them
# (a claim is reclaimable once its job leaves squeue, or after claim_stale_seconds).  An item with
# nothing outstanding exits before loading the engine, so the second sweep is cheap.  Per-document
# output depends only on (shard, row), never on which worker or item order produced it.
#
# Required exports: KYS_WORKLIST, KYS_CODE_COMMIT (the launcher's HEAD; a requeued task refuses to
# run different code than was submitted).
: "${KYS_WORKLIST:?set KYS_WORKLIST (setting:pass+setting:pass+...)}"
: "${KYS_CODE_COMMIT:?set KYS_CODE_COMMIT}"
CODE=/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data
# The configured `data` interpreter (absolute, as in smoke/pilot.sbatch) reads cluster.yaml and
# production.yaml -- never whatever `python3` happens to be on the submitting shell's PATH.
PY_DATA=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
[[ -x "$PY_DATA" ]] || { echo "[rewrite] data interpreter '$PY_DATA' is not executable" >&2; exit 2; }

# ---- code pin: requeued scavenger tasks re-read the checkout, so it must not have moved ----
HEAD_NOW=$(git -C "$CODE" rev-parse HEAD)
DIRTY=$(git -C "$CODE" status --porcelain | wc -l)
if [[ "$HEAD_NOW" != "$KYS_CODE_COMMIT" || "$DIRTY" != "0" ]]; then
  echo "[rewrite] code checkout is $HEAD_NOW (dirty=$DIRTY), submitted at $KYS_CODE_COMMIT -- refusing to run" >&2
  exit 2
fi

# ---- scope gate (fail closed): every item must be an allowed setting and a known pass ----
PY=$(PYTHONPATH="$CODE/src" KYS_WORKLIST="$KYS_WORKLIST" "$PY_DATA" - <<'PYX'
import os, sys
from kys3b.config import Config, production_scope
scope = production_scope()
for item in os.environ["KYS_WORKLIST"].split("+"):
    setting, _, pass_name = item.partition(":")
    if setting not in scope["allowed_settings"] or pass_name not in scope["passes"]:
        sys.exit(f"[rewrite] worklist item {item!r} is outside configs/production.yaml scope")
print(Config.load().cluster["env"]["python_vllm"])
PYX
)
[[ -x "$PY" ]] || { echo "[rewrite] configured python_vllm '$PY' is not executable" >&2; exit 2; }
echo "[rewrite] worklist=$KYS_WORKLIST python=$PY code_commit=$HEAD_NOW"
"$PY" -c "import vllm, torch, transformers; print('[rewrite] vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

IFS='+' read -r -a ITEMS <<< "$KYS_WORKLIST"
for SWEEP in 1 2; do
  for ITEM in "${ITEMS[@]}"; do
    export KYS_SETTING="${ITEM%%:*}" KYS_PASS="${ITEM##*:}"
    # The namespace MUST include the array job id.  The primary jhu2 array and the
    # opportunistic scavenger array both number their tasks from 0, so a task-id-only
    # namespace would let primary task 0 and scavenger task 0 share TMPDIR,
    # VLLM_CACHE_ROOT and TORCHINDUCTOR_CACHE_DIR while running concurrently.
    export KYS_NS="rewrite/${KYS_SETTING}/${KYS_PASS}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/${SLURM_ARRAY_TASK_ID:-0}"
    source "$CODE/slurm/_common.sh"
    echo "[rewrite] sweep=$SWEEP setting=$KYS_SETTING pass=$KYS_PASS $(date -Is)"
    srun "$PY" -m kys3b.rewrite.worker --setting "$KYS_SETTING" --pass "$KYS_PASS"
  done
done
echo "[rewrite] task finished $(date -Is)"
