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
# One array task = one GPU = one vLLM process (tensor_parallel_size=1), exactly as at 1.5B.
# `--gres=gpu:1` uses the GENERIC type so a single array can land on either h100 or h200;
# `--gres=gpu:h100:1` would never schedule on h200.
#
# account / qos / array / --requeue are supplied by bin/03_rewrite_launch.py:
#   primary       : -A bvandur1_blabscify -q jhu2      --array=0-31%32
#                   (jhu2 ALLOWS UP TO 32 concurrent GPUs; it does not guarantee them)
#   opportunistic : -A bvandur1           -q scavenger --array=0-63%64 --requeue
#                   (preemptible, PreemptMode=REQUEUE; no ceiling and no promise)
# Both queues run this same script against the SAME claim directory, so they cooperate.
# A worker exits cleanly when no shards remain, so resubmitting is the resume.
#
# Required exports: KYS_SETTING, KYS_PASS
: "${KYS_SETTING:?set KYS_SETTING}"
: "${KYS_PASS:?set KYS_PASS}"
# The namespace MUST include the array job id.  The primary jhu2 array and the
# opportunistic scavenger array both number their tasks from 0, so a task-id-only
# namespace would let primary task 0 and scavenger task 0 share TMPDIR,
# VLLM_CACHE_ROOT and TORCHINDUCTOR_CACHE_DIR while running concurrently.
export KYS_NS="rewrite/${KYS_SETTING}/${KYS_PASS}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/${SLURM_ARRAY_TASK_ID:-0}"
source /weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/slurm/_common.sh

PY=$(python3 - <<'PYX'
import yaml,pathlib
c=yaml.safe_load(pathlib.Path('/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/configs/cluster.yaml').read_text())
print(c['env']['python_vllm'])
PYX
)
echo "[slurm] setting=$KYS_SETTING pass=$KYS_PASS python=$PY"

srun "$PY" -m kys3b.rewrite.worker --setting "$KYS_SETTING" --pass "$KYS_PASS"
