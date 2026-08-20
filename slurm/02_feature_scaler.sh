#!/usr/bin/env bash
#SBATCH --job-name=gwl-scaler
#SBATCH --partition=cpu            # adjust to your cluster's CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/feature_scaler_%j.out
#SBATCH --error=logs/slurm/feature_scaler_%j.err

set -euo pipefail

echo "=== Job: feature_scaler | Node: $(hostname) | Start: $(date) ==="

# source .venv/bin/activate
source $HOME/.venv/bin/activate
export PYTHONUNBUFFERED=1
${PY:-python} -m gwlcore.feature_scaler \
    --data-dir "${DATA_DIR}"

# ── Append to experiment summary ───────────────────────────────────────
if [ -n "${SUMMARY_FILE:-}" ]; then
cat >> "${SUMMARY_FILE}" <<EOF

--- Feature Scaler (02) ---
data-dir:                   ${DATA_DIR}
EOF
fi

echo "=== Finished: $(date) ==="