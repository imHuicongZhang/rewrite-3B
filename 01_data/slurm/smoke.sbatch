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
# silently turn into a production run.  With KYS_SMOKE_ROWS set, output goes to
# <setting>/_smoke/<pass>/ and NEVER to the production path, so production still sees the shard
# as outstanding.
#
# Required exports: KYS_SETTING, KYS_PASS.  Optional: KYS_SMOKE_ROWS (default 64).
: "${KYS_SETTING:?set KYS_SETTING}"
: "${KYS_PASS:?set KYS_PASS}"
KYS_SMOKE_ROWS="${KYS_SMOKE_ROWS:-64}"
export KYS_NS="smoke/${KYS_SETTING}/${KYS_PASS}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/${SLURM_ARRAY_TASK_ID:-0}"
source /weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/slurm/_common.sh

PY=$(python3 - <<'PYX'
import yaml, pathlib
c = yaml.safe_load(pathlib.Path(
    '/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/configs/cluster.yaml'
).read_text())
print(c['env']['python_vllm'])
PYX
)
echo "[smoke] setting=$KYS_SETTING pass=$KYS_PASS rows=$KYS_SMOKE_ROWS python=$PY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" -m kys3b.rewrite.worker \
  --setting "$KYS_SETTING" --pass "$KYS_PASS" \
  --max-shards 1 --smoke-rows "$KYS_SMOKE_ROWS"
