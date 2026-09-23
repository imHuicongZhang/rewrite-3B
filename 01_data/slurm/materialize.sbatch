#!/usr/bin/env bash
#SBATCH --job-name=kys3b_materialize
#SBATCH --partition=med
#SBATCH --qos=jhu2          # association QOS; cpu_qos is med's PARTITION QoS and is not requestable
#SBATCH --account=bvandur1_blabscify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#
# Stage 02.  Optional $@ = settings.  Idempotent per shard.
#
# Sized for the batched materializer: one 64-shard batch buffers ~640k documents of text (a couple
# of GB) plus one pool shard's text column at a time.  Right-sized alongside select.sbatch so it
# schedules on a busy med partition.
export KYS_NS="materialize"
source /weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/slurm/_common.sh
PY=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
srun "$PY" "$CODE/bin/02_materialize_sources.py" ${@:+--settings "$@"}
