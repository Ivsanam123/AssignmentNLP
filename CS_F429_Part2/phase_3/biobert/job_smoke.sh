#!/bin/bash
#SBATCH --job-name=biobert_smoke
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=phase_3/biobert/logs/%x_%j.log

# Cheap environment check: 200 sentences, 1 epoch. Run this FIRST so a typo or
# a missing package costs 5 minutes of queue time instead of two hours.
# It writes no results files.

set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
cd "$ROOT"
mkdir -p phase_3/biobert/logs

source "$ROOT/.venv_gpu/bin/activate"
export HF_HOME="$ROOT/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo

python phase_3/biobert/biobert_baseline.py --smoke

echo "smoke test OK"
