#!/usr/bin/env bash
#SBATCH --job-name=kys3b_select
#SBATCH --partition=med
#SBATCH --qos=cpu_qos
#SBATCH --account=bvandur1_blabscify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#
# Stage 01, CPU only.  Pass the phase as $1: `report` writes nothing under 01_selection/.
#   sbatch slurm/select.sbatch report
#   sbatch slurm/select.sbatch commit
export KYS_NS="select"
source /weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/slurm/_common.sh
PHASE="${1:-report}"
PY=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
srun "$PY" "$CODE/bin/01_select.py" --phase "$PHASE"
