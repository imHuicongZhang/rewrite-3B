#!/usr/bin/env bash
#SBATCH --job-name=kys3b_final
#SBATCH --partition=med
#SBATCH --qos=cpu_qos
#SBATCH --account=bvandur1_blabscify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#
# Stage 05: base + strategy -> shuffled corpora.  Optional $@ = settings.
# --mem-gb must match --mem so bucketed_shuffle sizes its buckets correctly.
export KYS_NS="final"
source /weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/slurm/_common.sh
PY=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
srun "$PY" "$CODE/bin/05_assemble_final.py" --mem-gb 340 ${@:+--settings "$@"}
