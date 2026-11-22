#!/bin/bash
# One-time environment setup. RUN THIS ON THE LOGIN NODE, not via sbatch:
#     bash phase_3/biobert/setup_env.sh
#
# Compute nodes on most clusters have no outbound internet, so both pip and the
# HuggingFace model download have to happen here, on the login node. The job
# script then runs fully offline.

set -euo pipefail

# Resolve the CS_F429_Part2 root from this script's location
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# Keep the HF cache inside the project so the compute node can see it and we
# don't blow up the home-directory quota.
export HF_HOME="$ROOT/.hf_cache"

echo "== project root: $ROOT"

# Clusters differ; uncomment whichever applies. Check with: module avail python
# module load python/3.11
# module load cuda/12.4

if [ ! -d "$ROOT/.venv_gpu" ]; then
    echo "== creating .venv_gpu"
    python3 -m venv "$ROOT/.venv_gpu"
fi
source "$ROOT/.venv_gpu/bin/activate"

python -m pip install -U pip wheel

# cu121 wheels work on RTX Pro 6000; if the driver is older, swap to cu118.
echo "== installing torch"
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "== installing the rest"
pip install "transformers>=4.40" "tokenizers>=0.19" numpy

echo "== pre-fetching dmis-lab/biobert-base-cased-v1.1 into $HF_HOME"
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForTokenClassification
name = "dmis-lab/biobert-base-cased-v1.1"
AutoTokenizer.from_pretrained(name)
AutoModelForTokenClassification.from_pretrained(name, num_labels=5)
print("model cached OK")
PY

echo
echo "== setup done. Next:"
echo "   sbatch phase_3/biobert/job_smoke.sh     # ~5 min, verifies GPU + data"
echo "   sbatch phase_3/biobert/job_biobert.sh   # the real run"
