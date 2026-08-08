#!/usr/bin/env bash
#SBATCH --job-name=gwl-mlp-train
#SBATCH --partition=gpu            # adjust to your cluster's CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err

set -euo pipefail
echo "=== Job: train | Node: $(hostname) | GPU: $(nvidia-smi -L) | Start: $(date) ==="

# ── Model type (default: lstm) ────────────────────────────────────────
MODEL_TYPE="${MODEL_TYPE:-lstm}"
USE_STATIC_FEATURES="${USE_STATIC_FEATURES:-yes}"
USE_RAIN_TEMP="${USE_RAIN_TEMP:-no}"
echo "MODEL_TYPE=${MODEL_TYPE}  USE_STATIC_FEATURES=${USE_STATIC_FEATURES}  USE_RAIN_TEMP=${USE_RAIN_TEMP}  DATA_DIR=${DATA_DIR}  OUTPUT_DIR=${OUTPUT_DIR}  MLFLOW_DIR=${MLFLOW_DIR}"

# Optional flag to drop static features (keeps only well_depth)
STATIC_FEATURES_ARG=()
if [ "${USE_STATIC_FEATURES}" = "no" ]; then
    STATIC_FEATURES_ARG=(--no-static-features)
fi

RAIN_TEMP_ARG=()
if [ "${USE_RAIN_TEMP}" = "yes" ]; then
    RAIN_TEMP_ARG=(--rain-temp-only)
fi

USE_WEIGHTED_LOSS="${USE_WEIGHTED_LOSS:-yes}"
LOSS_WEIGHT_TYPE="${LOSS_WEIGHT_TYPE:-std}"   # std | perwell
LOSS_TRIM_HIGH="${LOSS_TRIM_HIGH:-0.0}"        # 0.0=off, 0.10=drop worst 10% wells/batch
WEIGHTED_LOSS_ARG=()
if [ "${USE_WEIGHTED_LOSS}" = "no" ]; then
    WEIGHTED_LOSS_ARG=(--no-weighted-loss)
else
    WEIGHTED_LOSS_ARG=(--loss-weight-type "${LOSS_WEIGHT_TYPE}" --loss-trim-high "${LOSS_TRIM_HIGH}")
fi

USE_REVIN="${USE_REVIN:-no}"
REVIN_ARG=()
if [ "${USE_REVIN}" = "yes" ]; then
    REVIN_ARG=(--use-revin)
fi
REVIN_STD_FLOOR="${REVIN_STD_FLOOR:-0.1}"
MIN_DELTA_FOR_TREND="${MIN_DELTA_FOR_TREND:-0.5}"

# Forecast horizon (months) — must match the HORIZON used by data prep so the
# default pred-clamp can be picked correctly (<=4mo → 0.20, >4mo → 0.50).
FORECAST_HORIZON_MONTHS="${HORIZON:-${FORECAST_HORIZON_MONTHS:-6}}"

# Inference-time prediction clamp. When unset, train.py picks the default from
# FORECAST_HORIZON_MONTHS. Set to 0.0 to disable; set a float (e.g. 0.30) to
# override the default explicitly.
PRED_CLAMP_PCT="${PRED_CLAMP_PCT:-}"
PRED_CLAMP_ARG=()
if [ -n "${PRED_CLAMP_PCT}" ]; then
    PRED_CLAMP_ARG=(--pred-clamp-pct "${PRED_CLAMP_PCT}")
fi

# Std knob (used for both data-prep flat filter and eval-time predictable
# cohort threshold). Comes from launch_pipeline.sh / 01_data_preparation.sh.
MIN_STATION_TARGET_STD="${MIN_STATION_TARGET_STD:-0.0}"
INCLUDE_STATES="${INCLUDE_STATES:-}"
MIN_STATION_MODE_GAP_DAYS="${MIN_STATION_MODE_GAP_DAYS:-0}"

# Optional additive linear baseline (residual-learning architecture).
USE_LINEAR_RESIDUAL="${USE_LINEAR_RESIDUAL:-no}"
LINEAR_RESIDUAL_ARG=()
if [ "${USE_LINEAR_RESIDUAL}" = "yes" ]; then
    LINEAR_RESIDUAL_ARG=(--use-linear-residual)
fi

# ── Prithvi-EO joint fine-tune (gated; default off → unchanged) ────────
# Requires MODEL_TYPE=tft and a tile_manifest.pkl in DATA_DIR (from 01 with
# USE_PRITHVI=yes). The frozen 300M base is loaded from PRITHVI_MODEL_DIR.
USE_PRITHVI="${USE_PRITHVI:-no}"
PRITHVI_ARG=()
if [ "${USE_PRITHVI}" = "yes" ]; then
    PRITHVI_ARG=(
        --use-prithvi
        --prithvi-model-dir "${PRITHVI_MODEL_DIR:-$HOME/prithvi_data_download/Prithvi-EO-2.0-300M-TL}"
        --station-index-csv "${STATION_INDEX_CSV:-/mnt/h200_disk/$USER/prithvi_gwl/gwl_stations.csv}"
        --prithvi-proj-dim "${PRITHVI_PROJ_DIM:-32}"
        --lora-r "${LORA_R:-16}"
        --lora-alpha "${LORA_ALPHA:-32}"
        --ft-lr "${FT_LR:-1e-4}"
        --forecaster-lr-scale "${FORECASTER_LR_SCALE:-10}"
        --prithvi-n-tiles "${PRITHVI_N_TILES:-256}"
        --prithvi-samples-per-tile "${PRITHVI_SAMPLES_PER_TILE:-32}"
    )
    if [ "${PRITHVI_GRAD_CKPT:-no}" = "yes" ]; then
        PRITHVI_ARG+=(--prithvi-grad-ckpt)
    fi
fi

# ── Model-specific args ──────────────────────────────────────────────
if [ "${MODEL_TYPE}" = "transformer" ]; then
    MODEL_ARGS=(
        --model-type transformer
        --transformer-d-model "${TRANSFORMER_D_MODEL:-64}"
        --transformer-nhead "${TRANSFORMER_NHEAD:-4}"
        --transformer-num-layers "${TRANSFORMER_NUM_LAYERS:-2}"
        --transformer-dim-feedforward "${TRANSFORMER_DIM_FF:-128}"
        --transformer-dropout "${TRANSFORMER_DROPOUT:-0.2}"
    )
elif [ "${MODEL_TYPE}" = "tft" ]; then
    MODEL_ARGS=(
        --model-type tft
        --tft-d-model "${TFT_D_MODEL:-64}"
        --tft-n-heads "${TFT_N_HEADS:-4}"
        --tft-lstm-layers "${TFT_LSTM_LAYERS:-1}"
        --tft-dropout "${TFT_DROPOUT:-0.1}"
    )
elif [ "${MODEL_TYPE}" = "conditioned_lstm" ]; then
    MODEL_ARGS=(
        --model-type conditioned_lstm
        --lstm-hidden-dim "${LSTM_HIDDEN_DIM:-256}"
        --lstm-num-layers "${LSTM_NUM_LAYERS:-2}"
        --lstm-dropout "${LSTM_DROPOUT:-0.2}"
    )
elif [ "${MODEL_TYPE}" = "conditioned_transformer" ]; then
    MODEL_ARGS=(
        --model-type conditioned_transformer
        --transformer-d-model "${TRANSFORMER_D_MODEL:-64}"
        --transformer-nhead "${TRANSFORMER_NHEAD:-4}"
        --transformer-num-layers "${TRANSFORMER_NUM_LAYERS:-2}"
        --transformer-dim-feedforward "${TRANSFORMER_DIM_FF:-128}"
        --transformer-dropout "${TRANSFORMER_DROPOUT:-0.2}"
    )
else
    MODEL_ARGS=(
        --model-type lstm
        --lstm-hidden-dim "${LSTM_HIDDEN_DIM:-256}"
        --lstm-num-layers "${LSTM_NUM_LAYERS:-2}"
        --lstm-dropout "${LSTM_DROPOUT:-0.2}"
    )
fi

# ── Shared architecture args ─────────────────────────────────────────
SHARED_ARGS=(
    --static-embedding-dim "${STATIC_EMBEDDING_DIM:-32}"
    --fusion-hidden-dim "${FUSION_HIDDEN_DIM:-64}"
    --dropout "${DROPOUT:-0.2}"
)

# ── Categorical embedding dims ───────────────────────────────────────
EMB_ARGS=(
    --lithology-emb-dim "${LITHOLOGY_EMB_DIM:-4}"
    --well-type-emb-dim "${WELL_TYPE_EMB_DIM:-3}"
    --aquifer-emb-dim "${AQUIFER_EMB_DIM:-2}"
    --aquifer-0-aquifer-emb-dim "${AQUIFER_0_AQUIFER_EMB_DIM:-3}"
    --litho-supergroup-emb-dim "${LITHO_SUPERGROUP_EMB_DIM:-3}"
    --lulc-emb-dim "${LULC_EMB_DIM:-3}"
    --state-emb-dim "${STATE_EMB_DIM:-4}"
    --district-emb-dim "${DISTRICT_EMB_DIM:-5}"
)

# ── Training hyperparams ─────────────────────────────────────────────
TRAIN_ARGS=(
    --epochs "${EPOCHS:-100}"
    --batch-size "${BATCH_SIZE:-16384}"
    --lr "${LR:-0.001}"
    --weight-decay "${WEIGHT_DECAY:-1e-5}"
    --gradient-clip-norm "${GRADIENT_CLIP_NORM:-1.0}"
    --huber-delta "${HUBER_DELTA:-1.0}"
    --alpha "${ALPHA:-0.7}"
    --beta "${BETA:-0.3}"
)

# ── Scheduler ─────────────────────────────────────────────────────────
SCHED_ARGS=(
    --scheduler-factor "${SCHEDULER_FACTOR:-0.5}"
    --scheduler-patience "${SCHEDULER_PATIENCE:-10}"
)

# ── Run ────────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source $HOME/.venv/bin/activate
RUN_NAME_ARG=()
if [ -n "${RUN_NAME:-}" ]; then
    RUN_NAME_ARG=(--run-name "${RUN_NAME}")
fi

${PY:-python} training/train.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "${RUN_NAME_ARG[@]}" \
    --device cuda \
    "${MODEL_ARGS[@]}" \
    "${SHARED_ARGS[@]}" \
    "${EMB_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SCHED_ARGS[@]}" \
    "${STATIC_FEATURES_ARG[@]}" \
    "${RAIN_TEMP_ARG[@]}" \
    "${WEIGHTED_LOSS_ARG[@]}" \
    "${REVIN_ARG[@]}" \
    "${LINEAR_RESIDUAL_ARG[@]}" \
    --min-station-target-std "${MIN_STATION_TARGET_STD}" \
    --include-states "${INCLUDE_STATES}" \
    --min-station-mode-gap-days "${MIN_STATION_MODE_GAP_DAYS}" \
    --min-delta-for-trend "${MIN_DELTA_FOR_TREND}" \
    --revin-std-floor "${REVIN_STD_FLOOR}" \
    --forecast-horizon-months "${FORECAST_HORIZON_MONTHS}" \
    "${PRED_CLAMP_ARG[@]}" \
    "${PRITHVI_ARG[@]}"

# ── Append to experiment summary ───────────────────────────────────────
if [ -n "${SUMMARY_FILE:-}" ]; then
cat >> "${SUMMARY_FILE}" <<EOF

--- Training (03) ---
model-type:                 ${MODEL_TYPE}
use-static-features:        ${USE_STATIC_FEATURES}
use-rain-temp:              ${USE_RAIN_TEMP}
use-weighted-loss:          ${USE_WEIGHTED_LOSS}
loss-weight-type:           ${LOSS_WEIGHT_TYPE}
loss-trim-high:             ${LOSS_TRIM_HIGH}
use-revin:                  ${USE_REVIN}
revin-std-floor:            ${REVIN_STD_FLOOR}
min-delta-for-trend:        ${MIN_DELTA_FOR_TREND}
min-station-target-std:     ${MIN_STATION_TARGET_STD}
include-states:             ${INCLUDE_STATES:-(none)}
min-station-mode-gap-days:  ${MIN_STATION_MODE_GAP_DAYS}
forecast-horizon-months:    ${FORECAST_HORIZON_MONTHS}
pred-clamp-pct:             ${PRED_CLAMP_PCT:-(auto from horizon)}
use-linear-residual:        ${USE_LINEAR_RESIDUAL}
use-prithvi:                ${USE_PRITHVI}
prithvi-model-dir:          ${PRITHVI_MODEL_DIR:-(n/a)}
prithvi-proj-dim:           ${PRITHVI_PROJ_DIM:-32}
lora-r/alpha:               ${LORA_R:-16}/${LORA_ALPHA:-32}
ft-lr:                      ${FT_LR:-1e-4}
forecaster-lr-scale:        ${FORECASTER_LR_SCALE:-10}
prithvi-n-tiles x spt:      ${PRITHVI_N_TILES:-256} x ${PRITHVI_SAMPLES_PER_TILE:-32}
prithvi-grad-ckpt:          ${PRITHVI_GRAD_CKPT:-no}
epochs:                     ${EPOCHS:-100}
batch-size:                 ${BATCH_SIZE:-16384}
lr:                         ${LR:-0.001}
weight-decay:               ${WEIGHT_DECAY:-1e-5}
gradient-clip-norm:         ${GRADIENT_CLIP_NORM:-1.0}
huber-delta:                ${HUBER_DELTA:-1.0}
alpha:                      ${ALPHA:-0.7}
beta:                       ${BETA:-0.3}
dropout:                    ${DROPOUT:-0.2}
static-embedding-dim:       ${STATIC_EMBEDDING_DIM:-32}
fusion-hidden-dim:          ${FUSION_HIDDEN_DIM:-64}
lithology-emb-dim:          ${LITHOLOGY_EMB_DIM:-4}
well-type-emb-dim:          ${WELL_TYPE_EMB_DIM:-3}
aquifer-emb-dim:            ${AQUIFER_EMB_DIM:-2}
aquifer-0-aquifer-emb-dim:  ${AQUIFER_0_AQUIFER_EMB_DIM:-3}
litho-supergroup-emb-dim:   ${LITHO_SUPERGROUP_EMB_DIM:-3}
lulc-emb-dim:               ${LULC_EMB_DIM:-3}
state-emb-dim:              ${STATE_EMB_DIM:-4}
district-emb-dim:           ${DISTRICT_EMB_DIM:-5}
scheduler-factor:           ${SCHEDULER_FACTOR:-0.5}
scheduler-patience:         ${SCHEDULER_PATIENCE:-10}
EOF

if [ "${MODEL_TYPE}" = "tft" ]; then
cat >> "${SUMMARY_FILE}" <<EOF
tft-d-model:                ${TFT_D_MODEL:-64}
tft-n-heads:                ${TFT_N_HEADS:-4}
tft-lstm-layers:            ${TFT_LSTM_LAYERS:-1}
tft-dropout:                ${TFT_DROPOUT:-0.1}
EOF
elif [ "${MODEL_TYPE}" = "transformer" ] || [ "${MODEL_TYPE}" = "conditioned_transformer" ]; then
cat >> "${SUMMARY_FILE}" <<EOF
transformer-d-model:        ${TRANSFORMER_D_MODEL:-64}
transformer-nhead:          ${TRANSFORMER_NHEAD:-4}
transformer-num-layers:     ${TRANSFORMER_NUM_LAYERS:-2}
transformer-dim-feedforward: ${TRANSFORMER_DIM_FF:-128}
transformer-dropout:        ${TRANSFORMER_DROPOUT:-0.2}
EOF
else
cat >> "${SUMMARY_FILE}" <<EOF
lstm-hidden-dim:            ${LSTM_HIDDEN_DIM:-256}
lstm-num-layers:            ${LSTM_NUM_LAYERS:-2}
lstm-dropout:               ${LSTM_DROPOUT:-0.2}
EOF
fi
fi

echo "=== Finished: $(date) ==="