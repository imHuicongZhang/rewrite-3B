#!/usr/bin/env bash
#SBATCH --job-name=kys3b_smoke
#SBATCH --partition=h100,h200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#
# BOUNDED GPU smoke test: ONE task, ONE shard, at most KYS_SMOKE_ROWS documents.
#
# Deliberately NOT an array and NOT the production wall time, so a smoke submission cannot
# silently turn into a production run.  With KYS_SMOKE_ROWS set, output, claims, progress and
# completion state all live under <setting>/_smoke/ and NEVER on the production paths, so
# production still sees the shard as outstanding and its totals exclude smoke tokens.
#
# Required exports: KYS_SETTING, KYS_PASS.  Optional: KYS_SMOKE_ROWS (default 64).
: "${KYS_SETTING:?set KYS_SETTING}"
: "${KYS_PASS:?set KYS_PASS}"
KYS_SMOKE_ROWS="${KYS_SMOKE_ROWS:-64}"
export KYS_NS="smoke/${KYS_SETTING}/${KYS_PASS}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/${SLURM_ARRAY_TASK_ID:-0}"
source /weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/slurm/_common.sh

# The vLLM interpreter comes from cluster.yaml env.python_vllm, read by the configured `data`
# interpreter (an absolute path, as in materialize.sbatch) -- never whatever `python3` happens to
# be on the submitting shell's PATH.
PY_DATA=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
PY=$("$PY_DATA" - <<'PYX'
import yaml, pathlib
c = yaml.safe_load(pathlib.Path(
    '/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/configs/cluster.yaml'
).read_text())
print(c['env']['python_vllm'])
PYX
)
[[ -x "$PY" ]] || { echo "[smoke] configured python_vllm '$PY' is not executable" >&2; exit 2; }
echo "[smoke] setting=$KYS_SETTING pass=$KYS_PASS rows=$KYS_SMOKE_ROWS python=$PY"
echo "[smoke] code_commit=$(git -C "$CODE" rev-parse HEAD) dirty=$(git -C "$CODE" status --porcelain | wc -l)"
"$PY" -c "import vllm, torch, transformers; print('[smoke] vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# Observation only: sample GPU memory every 2 s so peak usage can be reported.
GPU_MEM_LOG="$DATA/dataset/logs/smoke/gpumem_${SLURM_JOB_ID:-local}.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 2 \
  > "$GPU_MEM_LOG" 2>/dev/null &
GPU_MON=$!
trap 'kill $GPU_MON 2>/dev/null || true' EXIT

srun "$PY" -m kys3b.rewrite.worker \
  --setting "$KYS_SETTING" --pass "$KYS_PASS" \
  --max-shards 1 --smoke-rows "$KYS_SMOKE_ROWS"
