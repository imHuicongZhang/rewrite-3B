#!/usr/bin/env bash
#SBATCH --job-name=kys3b_post
#SBATCH --partition=med
#SBATCH --qos=jhu2          # association QOS; cpu_qos is med's PARTITION QoS and is not requestable
#SBATCH --account=bvandur1_blabscify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#
# Stage 04, ONE STEP PER JOB.  $1 = strip | score-rewire | assemble, $2.. = settings.
#
# One step per job is deliberate: the 1.5B run hit NODE_FAIL twice (jobs 1593189, 1610294)
# when a heavy 96-worker srun was followed by a second srun in the same batch job.  Chain the
# steps with --dependency=afterok instead.
export KYS_NS="post/${1:-step}"
source /weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/slurm/_common.sh
STEP="${1:?step}"; shift || true
PY=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
srun "$PY" "$CODE/bin/04_postprocess.py" --step "$STEP" ${@:+--settings "$@"}
