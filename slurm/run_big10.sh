#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_big10.sh — DIRECT (non-sbatch) pipeline launcher for the big-10 H200 node.
#
# Same configurability as slurm/launch_pipeline.sh, but:
#   • No #SBATCH headers — run it directly:  bash slurm/run_big10.sh
#   • Every knob is override-friendly (${VAR:-default}) so you can set any of
#     them inline per run, e.g.:
#         CUDA_VISIBLE_DEVICES=0 USE_PRITHVI=yes RUN_NAME=tft_prithvi_on bash slurm/run_big10.sh
#         CUDA_VISIBLE_DEVICES=1 USE_PRITHVI=no  RUN_NAME=tft_baseline    bash slurm/run_big10.sh
#   • GPU is pinned by CUDA_VISIBLE_DEVICES set OUTSIDE (this script never sets it).
#   • Defaults to the Prithvi-EO joint fine-tune (MODEL_TYPE=tft, USE_PRITHVI=yes,
#     half-yearly composites).
#   • The ENTIRE run (stdout+stderr) is tee'd to ${RUN_DIR}/pipeline.log.
#   • Outputs go under /mnt/h200_disk/$USER/gwl_lstm/output/prithvi/${RUN_NAME}.
#   • Optional REUSE_DATA_DIR=<dir>: skip data-prep+scaling and train against an
#     already-prepared data dir (so an ablation pair can share one prep).
#
# Launch 8 in parallel by varying CUDA_VISIBLE_DEVICES + RUN_NAME.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source ~/.bashrc 2>/dev/null || true

PIPELINE_START=$(date +%s)
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# ─────────────────────────────────────────────
# Run identity + directories
# ─────────────────────────────────────────────
CVD="${CUDA_VISIBLE_DEVICES:-unset}"
# ALWAYS timestamp the run dir so runs never overwrite each other:
#   RUN_NAME given  -> <RUN_NAME>_<timestamp>   (e.g. tft_prithvi_on_20260625_233045)
#   RUN_NAME unset  -> <timestamp>_gpu<id>
# (A provided RUN_NAME is a label, NOT a fixed dir — the timestamp guarantees uniqueness.)
if [ -n "${RUN_NAME:-}" ]; then
    RUN_NAME="${RUN_NAME}_${TIMESTAMP}"
else
    RUN_NAME="${TIMESTAMP}_gpu${CVD}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/h200_disk/$USER/gwl_lstm/output/prithvi}"
RUN_DIR="${OUTPUT_BASE}/${RUN_NAME}"
DATA_DIR="${RUN_DIR}/data"
PLOT_DIR="${RUN_DIR}/plots"
# train.py builds its run dir as OUTPUT_DIR/RUN_NAME, so point OUTPUT_DIR at the
# base → train writes directly into RUN_DIR (single level, alongside data/plots).
OUTPUT_DIR="${OUTPUT_BASE}"
MLFLOW_DIR="${MLFLOW_DIR:-$HOME/gwl_lstm/mlruns}"
export RUN_NAME

mkdir -p "${DATA_DIR}" "${PLOT_DIR}"

# ── Persist the FULL run to a log file (the thing slurm's --output did) ──
LOG_FILE="${RUN_DIR}/pipeline.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "═════════════════════════════════════════════════════════════════"
echo "run_big10.sh  |  $(date '+%Y-%m-%d %H:%M:%S')  |  host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CVD}"
echo "RUN_NAME=${RUN_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "═════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────
# Experiment config (edit defaults here; override any inline)
# ─────────────────────────────────────────────

# ── Model selection ──────────────────────────  (defaults mirror big-10 launch_pipeline.sh)
export MODEL_TYPE="${MODEL_TYPE:-tft}"                 # prithvi requires tft
export USE_STATIC_FEATURES="${USE_STATIC_FEATURES:-yes}"
export USE_RAIN_TEMP="${USE_RAIN_TEMP:-yes}"
export DROP_NDVI_SM="${DROP_NDVI_SM:-yes}"
export USE_ONLY_GWL="${USE_ONLY_GWL:-yes}"
export USE_WEIGHTED_LOSS="${USE_WEIGHTED_LOSS:-no}"
export LOSS_WEIGHT_TYPE="${LOSS_WEIGHT_TYPE:-none}"
export LOSS_TRIM_HIGH="${LOSS_TRIM_HIGH:-0.0}"
export USE_REVIN="${USE_REVIN:-yes}"
export REVIN_STD_FLOOR="${REVIN_STD_FLOOR:-0.1}"
export MIN_DELTA_FOR_TREND="${MIN_DELTA_FOR_TREND:-0.5}"
export PRED_CLAMP_PCT="${PRED_CLAMP_PCT:-0.6}"
export USE_LINEAR_RESIDUAL="${USE_LINEAR_RESIDUAL:-no}"

# ── Prithvi-EO joint fine-tune ───────────────
export USE_PRITHVI="${USE_PRITHVI:-yes}"
export COMPOSITE_DIR="${COMPOSITE_DIR:-/mnt/h200_disk/$USER/prithvi_gwl/half_yearly}"
export COMPOSITE_PERIOD="${COMPOSITE_PERIOD:-halfyear}"   # halfyear | quarter
export STATION_INDEX_CSV="${STATION_INDEX_CSV:-/mnt/h200_disk/$USER/prithvi_gwl/gwl_stations.csv}"
export PRITHVI_MODEL_DIR="${PRITHVI_MODEL_DIR:-$HOME/prithvi_data_download/Prithvi-EO-2.0-300M-TL}"
export PRITHVI_PROJ_DIM="${PRITHVI_PROJ_DIM:-32}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export FT_LR="${FT_LR:-1e-4}"                            # LoRA + projector LR
export FORECASTER_LR_SCALE="${FORECASTER_LR_SCALE:-10}"  # TFT LR = FT_LR * this
export PRITHVI_N_TILES="${PRITHVI_N_TILES:-64}"          # unique tiles / train batch
export PRITHVI_SAMPLES_PER_TILE="${PRITHVI_SAMPLES_PER_TILE:-32}"  # batch = n_tiles*spt = 2048 (matches BATCH_SIZE)
export PRITHVI_GRAD_CKPT="${PRITHVI_GRAD_CKPT:-no}"

# ── Data preparation (01) ────────────────────
export CSV_PATH="${CSV_PATH:-/mnt/h200_disk/$USER/csv_data/data_with_flag_all.csv}"
export HORIZON="${HORIZON:-3}"
export GAP_DAYS="${GAP_DAYS:-30}"
export LOOKBACK="${LOOKBACK:-1}"
export LOOKBACK_WINDOW="${LOOKBACK_WINDOW:-6d}"
export MAX_GWL="${MAX_GWL:-100.0}"
export MIN_GWL="${MIN_GWL:-0.0}"
export MIN_SEQ_COMPLETENESS="${MIN_SEQ_COMPLETENESS:-0.0}"
export MIN_STATION_SAMPLES="${MIN_STATION_SAMPLES:-0}"
export MIN_SAMPLE_FREQ_DAYS="${MIN_SAMPLE_FREQ_DAYS:-7}"
export MIN_SAMPLE_FREQ_DAYS_TRAIN="${MIN_SAMPLE_FREQ_DAYS_TRAIN:--1}"
export MIN_SAMPLE_FREQ_DAYS_VAL="${MIN_SAMPLE_FREQ_DAYS_VAL:--1}"
export MIN_SAMPLE_FREQ_DAYS_TEST="${MIN_SAMPLE_FREQ_DAYS_TEST:--1}"
export MAX_SAMPLES_PER_STATION_EVAL="${MAX_SAMPLES_PER_STATION_EVAL:-30}"
export MIN_STATION_TARGET_STD="${MIN_STATION_TARGET_STD:-0.0}"
export MAX_STATION_TARGET_STD="${MAX_STATION_TARGET_STD:-inf}"
export INCLUDE_STATES="${INCLUDE_STATES:-}"
export MIN_STATION_MODE_GAP_DAYS="${MIN_STATION_MODE_GAP_DAYS:-0}"
export INTERPOLATE_LOOKBACK_GWL="${INTERPOLATE_LOOKBACK_GWL:-yes}"
# Per-run shared-artifact dir (under RUN_DIR) so parallel preps don't clobber.
export SHARED_ARTIFACT_DIR="${SHARED_ARTIFACT_DIR:-${RUN_DIR}/shared}"
export TRAIN_END="${TRAIN_END:-2024-12-31}"
export VAL_START="${VAL_START:-2025-01-01}"
export VAL_END="${VAL_END:-2025-08-31}"
export TEST_START="${TEST_START:-2025-09-01}"
export SPLIT_STRATEGY="${SPLIT_STRATEGY:-station_time}"
export STATION_TRAIN_FRAC="${STATION_TRAIN_FRAC:-0.60}"
export STATION_VAL_FRAC="${STATION_VAL_FRAC:-0.25}"

# ── Training hyperparams (03) ────────────────
export ALPHA="${ALPHA:-1.0}"
export BETA="${BETA:-0.0}"
export EPOCHS="${EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-2048}"                  # val/test batching (train uses n_tiles*spt under prithvi)
export LR="${LR:-0.001}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
export GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
export HUBER_DELTA="${HUBER_DELTA:-1.0}"
export DROPOUT="${DROPOUT:-0.2}"

# ── Architecture: shared (03) ────────────────
export STATIC_EMBEDDING_DIM="${STATIC_EMBEDDING_DIM:-32}"
export FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM:-64}"

# ── Categorical embedding dims (03) ──────────
export LITHOLOGY_EMB_DIM="${LITHOLOGY_EMB_DIM:-3}"
export WELL_TYPE_EMB_DIM="${WELL_TYPE_EMB_DIM:-2}"
export AQUIFER_EMB_DIM="${AQUIFER_EMB_DIM:-2}"
export AQUIFER_0_AQUIFER_EMB_DIM="${AQUIFER_0_AQUIFER_EMB_DIM:-2}"
export LITHO_SUPERGROUP_EMB_DIM="${LITHO_SUPERGROUP_EMB_DIM:-2}"
export LULC_EMB_DIM="${LULC_EMB_DIM:-2}"
export STATE_EMB_DIM="${STATE_EMB_DIM:-3}"
export DISTRICT_EMB_DIM="${DISTRICT_EMB_DIM:-3}"

# ── Architecture: LSTM (03) ──────────────────
export LSTM_HIDDEN_DIM="${LSTM_HIDDEN_DIM:-256}"
export LSTM_NUM_LAYERS="${LSTM_NUM_LAYERS:-2}"
export LSTM_DROPOUT="${LSTM_DROPOUT:-0.2}"

# ── Architecture: Transformer (03) ───────────
export TRANSFORMER_D_MODEL="${TRANSFORMER_D_MODEL:-64}"
export TRANSFORMER_NHEAD="${TRANSFORMER_NHEAD:-4}"
export TRANSFORMER_NUM_LAYERS="${TRANSFORMER_NUM_LAYERS:-2}"
export TRANSFORMER_DIM_FF="${TRANSFORMER_DIM_FF:-128}"
export TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.2}"

# ── Architecture: TFT (03) ───────────────────
export TFT_D_MODEL="${TFT_D_MODEL:-64}"
export TFT_N_HEADS="${TFT_N_HEADS:-4}"
export TFT_LSTM_LAYERS="${TFT_LSTM_LAYERS:-1}"
export TFT_DROPOUT="${TFT_DROPOUT:-0.1}"

# ── Scheduler (03) ───────────────────────────
export SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.5}"
export SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-10}"

EXPERIMENT_COMMENT="${EXPERIMENT_COMMENT:-${MODEL_TYPE} prithvi=${USE_PRITHVI}}"

export DATA_DIR PLOT_DIR OUTPUT_DIR MLFLOW_DIR

# ─────────────────────────────────────────────
# Experiment summary header
# ─────────────────────────────────────────────
export SUMMARY_FILE="${RUN_DIR}/experiment_summary.txt"
cat > "${SUMMARY_FILE}" <<EOF
Experiment Summary
==================
Date:       $(date '+%Y-%m-%d %H:%M:%S')
Node:       $(hostname)
GPU (CUDA_VISIBLE_DEVICES): ${CVD}
Run Dir:    ${RUN_DIR}

Comment: ${EXPERIMENT_COMMENT:-"(none)"}
EOF
echo "Experiment summary: ${SUMMARY_FILE}"

# ─────────────────────────────────────────────
# Timer
# ─────────────────────────────────────────────
run_step () {
    NAME="$1"; CMD="$2"
    echo "─────────────────────────────────────────────"
    echo "Starting: $NAME   ($(date '+%Y-%m-%d %H:%M:%S'))"
    START=$(date +%s)
    eval "$CMD"
    DURATION=$(( $(date +%s) - START ))
    printf "Finished: %s   Duration: %02d:%02d:%02d\n" \
        "$NAME" $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60))
}

# ─────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────
if [ -n "${REUSE_DATA_DIR:-}" ]; then
    # Reuse an already-prepared data dir (skip prep + scaling). Useful so an
    # ablation pair (prithvi on/off) shares ONE prepared dataset.
    if [ ! -f "${REUSE_DATA_DIR}/train_samples.pkl" ]; then
        echo "ERROR: REUSE_DATA_DIR=${REUSE_DATA_DIR} has no train_samples.pkl" >&2
        exit 1
    fi
    export DATA_DIR="${REUSE_DATA_DIR}"
    echo "REUSE_DATA_DIR set → skipping data-prep + scaling; DATA_DIR=${DATA_DIR}"
    if [ "${USE_PRITHVI}" = "yes" ] && [ ! -f "${DATA_DIR}/tile_manifest.pkl" ]; then
        echo "WARNING: USE_PRITHVI=yes but ${DATA_DIR}/tile_manifest.pkl missing — "\
             "that data dir was prepared without --use-prithvi." >&2
    fi
else
    run_step "Data Preparation" "bash slurm/01_data_preparation.sh"
    run_step "Feature Scaling"  "bash slurm/02_feature_scaler.sh"
fi

run_step "Training" "bash slurm/03_train.sh"

# ─────────────────────────────────────────────
TOTAL=$(( $(date +%s) - PIPELINE_START ))
echo "─────────────────────────────────────────────"
echo "Pipeline finished. $(date '+%Y-%m-%d %H:%M:%S')"
printf "Total time: %02d:%02d:%02d\n" $((TOTAL/3600)) $((TOTAL%3600/60)) $((TOTAL%60))
echo "Outputs:  ${RUN_DIR}"
echo "Log:      ${LOG_FILE}"
