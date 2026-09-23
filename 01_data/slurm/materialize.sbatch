#!/usr/bin/env bash
#SBATCH --job-name=kys3b_materialize
#SBATCH --partition=med
#SBATCH --qos=jhu2          # association QOS; cpu_qos is med's PARTITION QoS and is not requestable
#SBATCH --account=bvandur1_blabscify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#
# Stage 02.  Optional $@ = settings.  Idempotent per shard.
export KYS_NS="materialize"
source /weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/slurm/_common.sh
PY=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
srun "$PY" "$CODE/bin/02_materialize_sources.py" ${@:+--settings "$@"}
