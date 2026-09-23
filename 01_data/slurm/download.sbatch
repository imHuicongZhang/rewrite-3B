#!/usr/bin/env bash
#SBATCH --job-name=kys3b_download
#SBATCH --partition=med
#SBATCH --qos=cpu_qos
#SBATCH --account=bvandur1_blabscify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#
# Stage 00: 182 GB from a GATED repo -- needs the HF token, so HF_HUB_OFFLINE is cleared here
# and ONLY here.
export KYS_NS="download"
source /weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite-3B/01_data/slurm/_common.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
PY=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
srun "$PY" "$CODE/bin/00_download_pool.py" --workers 8
