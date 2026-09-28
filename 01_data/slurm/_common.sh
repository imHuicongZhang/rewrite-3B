# Shared environment for every kys3b Slurm job.  Sourced, never executed.
#
# Caches must NEVER go to $HOME (the 1.5B lesson: home quota).  They are namespaced per
# (stage, setting, pass, array task) so concurrent torch.compile cannot collide.
set -euo pipefail

CODE=/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data
DATA=/weka/projects/bvandur1/zhuicon1/rewrite-3b
CACHE=$DATA/.cache
NS="${KYS_NS:-generic}"

# --export=ALL (the sbatch default) propagates the SUBMITTING shell's SLURM_CPUS_PER_TASK into
# this job.  If the submitter is itself inside an allocation with a different cpus-per-task, srun
# aborts: "cpus-per-task set by two different environment variables SLURM_CPUS_PER_TASK=17 !=
# SLURM_TRES_PER_TASK=cpu=16".  Normalise to THIS job's allocation so the two agree, and keep
# SLURM_CPUS_PER_TASK meaningful -- kys3b reads it to size its worker pools.
if [[ -n "${SLURM_TRES_PER_TASK:-}" ]]; then
  _KYS_CPUS=$(sed -n 's/.*cpu=\([0-9][0-9]*\).*/\1/p' <<< "$SLURM_TRES_PER_TASK")
  if [[ -n "$_KYS_CPUS" ]]; then
    export SLURM_CPUS_PER_TASK="$_KYS_CPUS"
    export SRUN_CPUS_PER_TASK="$_KYS_CPUS"
  fi
fi

export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export PIP_CACHE_DIR=$CACHE/pip
export HF_HOME=$CACHE/huggingface
export XDG_CACHE_HOME=$CACHE/xdg
export TMPDIR=$CACHE/tmp/$NS
export VLLM_CACHE_ROOT=$CACHE/vllm/$NS
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor/$NS
# FlashInfer JIT-compiles its top-k/top-p sampler with the NODE's /usr/local/cuda into
# $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer, defaulting to $HOME.  The GPU nodes do not share
# one CUDA toolkit: a sampling.so built during smoke links libcudart.so.12 and cannot load on
# gh126 (driver 595.91), so pilot jobs 915436/915437 died at engine start.  Every request here is
# greedy (temperature 0), and vLLM's Sampler returns argmax for an all-greedy batch before the
# top-k/top-p sampler is reached -- that kernel only ever runs in the engine's warm-up dummy pass.
# Use the PyTorch-native sampler (no JIT, no node toolkit), and keep any other FlashInfer JIT
# out of $HOME and per job.  Generated tokens are unaffected.
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer/$NS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$FLASHINFER_WORKSPACE_BASE"

# vLLM binds its ZMQ IPC sockets at $VLLM_RPC_BASE_PATH/<36-char uuid>, defaulting to TMPDIR.  A
# Unix socket path is capped at 107 bytes, and the namespaced TMPDIR above already exceeds that
# once the uuid is appended (all five smoke jobs 915188-915192 died at engine start with
# "ipc path ... is longer than 107 characters").  Sockets only, so a short node-local per-job
# directory is safe; TMPDIR and every cache stay namespaced as before.
export VLLM_RPC_BASE_PATH="/tmp/kys3b_${SLURM_JOB_ID:-$$}"
if (( ${#VLLM_RPC_BASE_PATH} + 37 > 107 )); then
  echo "[slurm] VLLM_RPC_BASE_PATH '$VLLM_RPC_BASE_PATH' too long for a Unix socket path" >&2
  exit 2
fi
mkdir -p "$VLLM_RPC_BASE_PATH"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-na} ns=$NS"
echo "[slurm] cpus_per_task=${SLURM_CPUS_PER_TASK:-?} tres_per_task=${SLURM_TRES_PER_TASK:-none}"
echo "[slurm] gpus=${CUDA_VISIBLE_DEVICES:-none} cpus=${SLURM_CPUS_PER_TASK:-?} restart=${SLURM_RESTART_COUNT:-0}"
echo "[slurm] vllm_rpc_base_path=$VLLM_RPC_BASE_PATH flashinfer_sampler=$VLLM_USE_FLASHINFER_SAMPLER flashinfer_workspace=$FLASHINFER_WORKSPACE_BASE"
