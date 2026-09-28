#!/usr/bin/env bash
#SBATCH --job-name=kys3b_pilot
#SBATCH --partition=h100,h200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#
# CALIBRATION PILOT: a small set of COMPLETE shards, spread across the whole selected corpus,
# with production generation semantics.  This is the yield gate before thousands of GPU-hours.
#
# Bounded by construction: the worker restricts itself to the K deterministic pilot shard ids
# (evenly spaced quantiles of the shard range -- NOT the first K, which are a biased slice of a
# doc_id-sorted selection).  Output goes to <setting>/_pilot/<pass>/ unless
# KYS_PILOT_TO_PRODUCTION=1, so it is trivially discardable if the pilot says something must
# change.  Claims, progress and the monitor are pilot-only too (kys3b.rewrite.worker.run_paths).  Not an array; a smaller wall time than production.
#
# Required exports: KYS_SETTING, KYS_PASS.  Optional: KYS_PILOT_K (default 12),
#                   KYS_PILOT_TO_PRODUCTION (default unset).
: "${KYS_SETTING:?set KYS_SETTING}"
: "${KYS_PASS:?set KYS_PASS}"
KYS_PILOT_K="${KYS_PILOT_K:-12}"
export KYS_NS="pilot/${KYS_SETTING}/${KYS_PASS}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/${SLURM_ARRAY_TASK_ID:-0}"
source /weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/slurm/_common.sh

# The vLLM interpreter comes from cluster.yaml env.python_vllm, read by the configured `data`
# interpreter (an absolute path, as in smoke.sbatch) -- never whatever `python3` happens to be on
# the submitting shell's PATH.
PY_DATA=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
PY=$("$PY_DATA" - <<'PYX'
import yaml, pathlib
c = yaml.safe_load(pathlib.Path(
    '/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/configs/cluster.yaml'
).read_text())
print(c['env']['python_vllm'])
PYX
)
[[ -x "$PY" ]] || { echo "[pilot] configured python_vllm '$PY' is not executable" >&2; exit 2; }
EXTRA=()
if [[ "${KYS_PILOT_TO_PRODUCTION:-0}" == "1" ]]; then EXTRA+=(--pilot-to-production); fi
echo "[pilot] setting=$KYS_SETTING pass=$KYS_PASS K=$KYS_PILOT_K to_production=${KYS_PILOT_TO_PRODUCTION:-0} python=$PY"
echo "[pilot] code_commit=$(git -C "$CODE" rev-parse HEAD) dirty=$(git -C "$CODE" status --porcelain | wc -l)"
"$PY" -c "import vllm, torch, transformers; print('[pilot] vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# Observation only: sample GPU memory every 10 s so peak usage can be reported.
GPU_MEM_LOG="$DATA/dataset/logs/pilot/gpumem_${SLURM_JOB_ID:-local}.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 10 \
  > "$GPU_MEM_LOG" 2>/dev/null &
GPU_MON=$!
trap 'kill $GPU_MON 2>/dev/null || true' EXIT

srun "$PY" -m kys3b.rewrite.worker \
  --setting "$KYS_SETTING" --pass "$KYS_PASS" --pilot "$KYS_PILOT_K" "${EXTRA[@]}"
