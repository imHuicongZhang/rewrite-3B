#!/usr/bin/env bash
#SBATCH --job-name=kys3b_download
#SBATCH --partition=med
#SBATCH --qos=jhu2          # association QOS; cpu_qos is med's PARTITION QoS and is not requestable
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

# _common.sh redirects HF_HOME onto the project filesystem (home has a quota), and
# huggingface_hub looks for the token at $HF_HOME/token -- so without this the job cannot
# authenticate to the GATED pool repo and the download fails.  Pass the token through the
# environment instead of copying it anywhere; never echo it.
if [[ -z "${HF_TOKEN:-}" && ! -s "$HF_HOME/token" && -s "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[download] HF token present (${#HF_TOKEN} chars)"
else
  echo "[download] WARNING: no HF token found; the gated repo will refuse the download" >&2
fi
PY=/weka/scratch/jhu/bvandur1/zhuicon1/envs/data/bin/python
srun "$PY" "$CODE/bin/00_download_pool.py" --workers 8
