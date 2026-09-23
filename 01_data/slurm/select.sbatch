#!/usr/bin/env bash
#SBATCH --job-name=kys3b_select
#SBATCH --partition=med
#SBATCH --qos=jhu2          # association QOS; cpu_qos is med's PARTITION QoS and is not requestable
#SBATCH --account=bvandur1_blabscify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# Right-sized from measurement, not guesswork.  The report run (job 850281) peaked at 13.2 GB RSS
# with 0.027 s of TotalCPU: PASS1's parquet reads are the only parallel part and finish in seconds,
# and the selection itself is single-threaded numpy.  A 96-CPU / 360 G ask was ~27x the real
# footprint and sat PENDING behind 199 other med jobs with an estimated start a day out.  24 CPUs
# and 96 G is ~7x headroom on the measured peak and schedules immediately.  Worker count cannot
# change the result: PASS1 writes each shard's slice into a preallocated array by offset, so the
# selection is identical for any number of workers.
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
