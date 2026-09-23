# Shared environment for every kys3b Slurm job.  Sourced, never executed.
#
# Caches must NEVER go to $HOME (the 1.5B lesson: home quota).  They are namespaced per
# (stage, setting, pass, array task) so concurrent torch.compile cannot collide.
set -euo pipefail

CODE=/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data
DATA=/weka/projects/bvandur1/zhuicon1/rewrite-3b
CACHE=$DATA/.cache
NS="${KYS_NS:-generic}"

export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export PIP_CACHE_DIR=$CACHE/pip
export HF_HOME=$CACHE/huggingface
export XDG_CACHE_HOME=$CACHE/xdg
export TMPDIR=$CACHE/tmp/$NS
export VLLM_CACHE_ROOT=$CACHE/vllm/$NS
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor/$NS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-na} ns=$NS"
echo "[slurm] gpus=${CUDA_VISIBLE_DEVICES:-none} cpus=${SLURM_CPUS_PER_TASK:-?} restart=${SLURM_RESTART_COUNT:-0}"
