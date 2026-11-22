#!/bin/bash
#SBATCH --job-name=biobert_ner
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=phase_3/biobert/logs/%x_%j.log

# Do NOT change --qos or remove --gres=mps:50 - this partition only
# accepts mps jobs under this QOS. Requesting --gres=gpu:1 or a
# different QOS here will be rejected at submission.

set -euo pipefail

# sbatch starts in the directory you submitted from; submit from CS_F429_Part2.
ROOT="$SLURM_SUBMIT_DIR"
cd "$ROOT"
mkdir -p phase_3/biobert/logs

source "$ROOT/.venv_gpu/bin/activate"

# Everything was pre-fetched by setup_env.sh on the login node; stay offline so
# a missing network on the compute node fails loudly instead of hanging.
export HF_HOME="$ROOT/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "job $SLURM_JOB_ID on $(hostname)   started $(date)"
nvidia-smi || true
echo

# batch 32 x max_len 256 fits comfortably in half an RTX Pro 6000.
python phase_3/biobert/biobert_baseline.py \
    --epochs 5 \
    --lr 3e-5 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --max-len 256 \
    --seed 42

echo
echo "finished $(date)"
