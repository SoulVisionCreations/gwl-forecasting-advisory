#!/usr/bin/env bash
#SBATCH --job-name=gwl-mlp-data-prep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --oversubscribe
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/data_preparation_%j.out
#SBATCH --error=logs/slurm/data_preparation_%j.err

set -euo pipefail
echo "=== Job: data_preparation | Node: $(hostname) | Start: $(date) ==="

# ── Data preparation params ────────────────────────────────────────────
CSV_PATH="${CSV_PATH:-/mnt/h200_disk/$USER/csv_data/data_with_flag_all.csv}"
HORIZON="${HORIZON:-3}"
GAP_DAYS="${GAP_DAYS:-30}"
LOOKBACK="${LOOKBACK:-5}"
# Parse LOOKBACK suffix: "5" → 5 years (months=0); "6m" → 6 months (years=0)
if [[ "${LOOKBACK}" == *m ]]; then
    LOOKBACK_YEARS_VAL="0"
    LOOKBACK_MONTHS_VAL="${LOOKBACK%m}"
else
    LOOKBACK_YEARS_VAL="${LOOKBACK}"
    LOOKBACK_MONTHS_VAL="0"
fi
LOOKBACK_WINDOW="${LOOKBACK_WINDOW:-3}"
MAX_GWL="${MAX_GWL:-400.0}"
MIN_GWL="${MIN_GWL:-0.0}"
MIN_SEQ_COMPLETENESS="${MIN_SEQ_COMPLETENESS:-0.5}"
MIN_STATION_SAMPLES="${MIN_STATION_SAMPLES:-0}"
MIN_SAMPLE_FREQ_DAYS="${MIN_SAMPLE_FREQ_DAYS:-14}"
MIN_SAMPLE_FREQ_DAYS_TRAIN="${MIN_SAMPLE_FREQ_DAYS_TRAIN:--1}"   # -1 inherit, 0 disable
MIN_SAMPLE_FREQ_DAYS_VAL="${MIN_SAMPLE_FREQ_DAYS_VAL:--1}"
MIN_SAMPLE_FREQ_DAYS_TEST="${MIN_SAMPLE_FREQ_DAYS_TEST:--1}"
MAX_SAMPLES_PER_STATION_EVAL="${MAX_SAMPLES_PER_STATION_EVAL:-30}"
MIN_STATION_TARGET_STD="${MIN_STATION_TARGET_STD:-0.0}"
MAX_STATION_TARGET_STD="${MAX_STATION_TARGET_STD:-inf}"
INCLUDE_STATES="${INCLUDE_STATES:-}"
MIN_STATION_MODE_GAP_DAYS="${MIN_STATION_MODE_GAP_DAYS:-0}"
INTERPOLATE_LOOKBACK_GWL="${INTERPOLATE_LOOKBACK_GWL:-no}"
INTERP_FLAG=()
if [ "${INTERPOLATE_LOOKBACK_GWL}" = "yes" ]; then
    INTERP_FLAG=(--interpolate-lookback-gwl)
fi
SHARED_ARTIFACT_DIR="${SHARED_ARTIFACT_DIR:-$HOME/shared}"
TRAIN_END="${TRAIN_END:-2024-12-31}"
VAL_START="${VAL_START:-2025-01-01}"
VAL_END="${VAL_END:-2025-08-31}"
TEST_START="${TEST_START:-2025-09-01}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-district}"
STATION_TRAIN_FRAC="${STATION_TRAIN_FRAC:-0.60}"
STATION_VAL_FRAC="${STATION_VAL_FRAC:-0.25}"
USE_RAIN_TEMP="${USE_RAIN_TEMP:-no}"
DROP_NDVI_SM="${DROP_NDVI_SM:-yes}"
USE_ONLY_GWL="${USE_ONLY_GWL:-no}"

# ── Prithvi-EO tile_idx annotation (gated; default off → unchanged) ─────
USE_PRITHVI="${USE_PRITHVI:-no}"
COMPOSITE_DIR="${COMPOSITE_DIR:-}"
COMPOSITE_PERIOD="${COMPOSITE_PERIOD:-halfyear}"
STATION_INDEX_CSV="${STATION_INDEX_CSV:-}"

echo "DATA_DIR=${DATA_DIR}  PLOT_DIR=${PLOT_DIR}"

# ── Run ────────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
source $HOME/.venv/bin/activate

RAIN_TEMP_ARG=()
if [ "${USE_RAIN_TEMP}" = "yes" ]; then
    RAIN_TEMP_ARG=(--rain-temp-only)
fi

DROP_FLAG=()
if [ "${DROP_NDVI_SM}" = "yes" ]; then
    DROP_FLAG=(--drop-ndvi-sm)
else
    DROP_FLAG=(--no-drop-ndvi-sm)
fi

ONLY_GWL_ARG=()
if [ "${USE_ONLY_GWL}" = "yes" ]; then
    ONLY_GWL_ARG=(--use-only-gwl)
fi

# Prithvi: annotate each sample with tile_idx + write tile_manifest.pkl.
PRITHVI_ARG=()
if [ "${USE_PRITHVI}" = "yes" ]; then
    PRITHVI_ARG=(
        --use-prithvi
        --composite-dir "${COMPOSITE_DIR}"
        --composite-period "${COMPOSITE_PERIOD}"
        --station-index-csv "${STATION_INDEX_CSV}"
    )
fi

${PY:-python} -m gwlcore.data_preparation \
    --csv-path "${CSV_PATH}" \
    --db-host "${GWL_DB_HOST:-localhost}" \
    --db-name "${GWL_DB_NAME:-gwl}" \
    --db-user "${GWL_DB_USER:-others}" \
    --db-password "${GWL_DB_PASSWORD:-}" \
    --horizon "${HORIZON}" \
    --gap-days "${GAP_DAYS}" \
    --output-dir "${DATA_DIR}" \
    --plot-dir "${PLOT_DIR}" \
    --lookback "${LOOKBACK_YEARS_VAL}" \
    --lookback-months "${LOOKBACK_MONTHS_VAL}" \
    --max-gwl "${MAX_GWL}" \
    --min-gwl "${MIN_GWL}" \
    --min-sequence-completeness "${MIN_SEQ_COMPLETENESS}" \
    --shared-artifact-dir "${SHARED_ARTIFACT_DIR}" \
    --train-end "${TRAIN_END}" \
    --val-start "${VAL_START}" \
    --val-end "${VAL_END}" \
    --test-start "${TEST_START}" \
    --split-strategy "${SPLIT_STRATEGY}" \
    --station-train-frac "${STATION_TRAIN_FRAC}" \
    --station-val-frac "${STATION_VAL_FRAC}" \
    "${RAIN_TEMP_ARG[@]}" \
    "${DROP_FLAG[@]}" \
    "${ONLY_GWL_ARG[@]}" \
    --lookback-window "${LOOKBACK_WINDOW}" \
    --min-station-samples "${MIN_STATION_SAMPLES}" \
    --min-sample-freq-days "${MIN_SAMPLE_FREQ_DAYS}" \
    --min-sample-freq-days-train "${MIN_SAMPLE_FREQ_DAYS_TRAIN}" \
    --min-sample-freq-days-val "${MIN_SAMPLE_FREQ_DAYS_VAL}" \
    --min-sample-freq-days-test "${MIN_SAMPLE_FREQ_DAYS_TEST}" \
    --max-samples-per-station-eval "${MAX_SAMPLES_PER_STATION_EVAL}" \
    --min-station-target-std "${MIN_STATION_TARGET_STD}" \
    --max-station-target-std "${MAX_STATION_TARGET_STD}" \
    --include-states "${INCLUDE_STATES}" \
    --min-station-mode-gap-days "${MIN_STATION_MODE_GAP_DAYS}" \
    "${INTERP_FLAG[@]}" \
    "${PRITHVI_ARG[@]}"

# ── Append to experiment summary ───────────────────────────────────────
if [ -n "${SUMMARY_FILE:-}" ]; then
cat >> "${SUMMARY_FILE}" <<EOF

--- Data Preparation (01) ---
csv-path:                   ${CSV_PATH}
horizon:                    ${HORIZON}
gap-days:                   ${GAP_DAYS}
lookback:                   ${LOOKBACK}
lookback-window:            ${LOOKBACK_WINDOW}
max-gwl:                    ${MAX_GWL}
min-gwl:                    ${MIN_GWL}
min-sequence-completeness:  ${MIN_SEQ_COMPLETENESS}
train-end:                  ${TRAIN_END}
val-start:                  ${VAL_START}
val-end:                    ${VAL_END}
test-start:                 ${TEST_START}
split-strategy:             ${SPLIT_STRATEGY}
use-rain-temp:              ${USE_RAIN_TEMP}
drop-ndvi-sm:               ${DROP_NDVI_SM}
use-only-gwl:               ${USE_ONLY_GWL}
min-sample-freq-days:       ${MIN_SAMPLE_FREQ_DAYS}  (train=${MIN_SAMPLE_FREQ_DAYS_TRAIN}, val=${MIN_SAMPLE_FREQ_DAYS_VAL}, test=${MIN_SAMPLE_FREQ_DAYS_TEST})
max-samples-per-station-eval: ${MAX_SAMPLES_PER_STATION_EVAL}
min-station-target-std:     ${MIN_STATION_TARGET_STD}
max-station-target-std:     ${MAX_STATION_TARGET_STD}
include-states:             ${INCLUDE_STATES:-(none)}
min-station-mode-gap-days:  ${MIN_STATION_MODE_GAP_DAYS}
interpolate-lookback-gwl:   ${INTERPOLATE_LOOKBACK_GWL}
use-prithvi:                ${USE_PRITHVI}
composite-dir:              ${COMPOSITE_DIR:-(none)}
composite-period:           ${COMPOSITE_PERIOD}
station-index-csv:          ${STATION_INDEX_CSV:-(sanitize fallback)}
EOF
fi

echo "=== Finished: $(date) ==="

