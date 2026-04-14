#!/usr/bin/env bash
# run_benchmark.sh — Run multiple models sequentially for a given task.
#
# Usage:
#   bash run_benchmark.sh [OPTIONS]
#
# Options:
#   --task          Task type: classification | segmentation | detection  (required)
#   --data_name     Dataset name, e.g. domars16k                          (required)
#   --repo_id       HuggingFace repo id, e.g. "Mirali33/mb-domars16k"    (required if load_from_hf=true)
#   --models        Space-separated model list, e.g. "resnet101 vit"      (default: all models for the task)
#   --training_type scratch_training | transfer_learning | feature_extraction (default: transfer_learning)
#   --epochs        Max epochs                                             (default: 50)
#   --lr            Learning rate                                          (default: 0.001)
#   --batch_size    Batch size                                             (default: 32)
#   --num_workers   Dataloader workers                                     (default: 4)
#   --load_from_hf  true | false                                           (default: true)
#   --wandb_project WandB project name                                     (default: marsbench-<task>)
#   --extra         Extra hydra overrides passed through verbatim          (default: "")
#
# Examples:
#   # Run all classification models on domars16k
#   bash run_benchmark.sh --task classification --data_name domars16k --repo_id "Mirali33/mb-domars16k"
#
#   # Run specific models with custom settings
#   bash run_benchmark.sh \
#       --task segmentation \
#       --data_name conequest_segmentation \
#       --repo_id "Mirali33/mb-conequest_seg" \
#       --models "unet mask2former" \
#       --epochs 100 \
#       --lr 0.0005
#
#   # Detection with feature extraction
#   bash run_benchmark.sh \
#       --task detection \
#       --data_name boulder_detection \
#       --repo_id "Mirali33/mb-boulder_det" \
#       --models "fasterrcnn ssd" \
#       --training_type feature_extraction

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
TASK=""
DATA_NAME=""
REPO_ID=""
MODELS=""
TRAINING_TYPE="transfer_learning"
EPOCHS=50
LR=0.001
BATCH_SIZE=32
NUM_WORKERS=0
LOAD_FROM_HF=true
WANDB_PROJECT=""
EXTRA=""

# ── model presets per task ───────────────────────────────────────────────────
ALL_CLASSIFICATION_MODELS="resnet101 inceptionv3 squeezenet swin_transformer vit"
ALL_SEGMENTATION_MODELS="unet deeplab segformer transformersDPT mask2former"
ALL_DETECTION_MODELS="fasterrcnn retinanet ssd"

# ── argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          TASK="$2";          shift 2 ;;
        --data_name)     DATA_NAME="$2";     shift 2 ;;
        --repo_id)       REPO_ID="$2";       shift 2 ;;
        --models)        MODELS="$2";        shift 2 ;;
        --training_type) TRAINING_TYPE="$2"; shift 2 ;;
        --epochs)        EPOCHS="$2";        shift 2 ;;
        --lr)            LR="$2";            shift 2 ;;
        --batch_size)    BATCH_SIZE="$2";    shift 2 ;;
        --num_workers)   NUM_WORKERS="$2";   shift 2 ;;
        --load_from_hf)  LOAD_FROM_HF="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --extra)         EXTRA="$2";         shift 2 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ── validate required args ───────────────────────────────────────────────────
if [[ -z "$TASK" || -z "$DATA_NAME" ]]; then
    echo "[ERROR] --task and --data_name are required."
    exit 1
fi

if [[ "$LOAD_FROM_HF" == "true" && -z "$REPO_ID" ]]; then
    echo "[ERROR] --repo_id is required when --load_from_hf=true."
    exit 1
fi

# ── resolve model list ───────────────────────────────────────────────────────
if [[ -z "$MODELS" ]]; then
    case "$TASK" in
        classification) MODELS="$ALL_CLASSIFICATION_MODELS" ;;
        segmentation)   MODELS="$ALL_SEGMENTATION_MODELS" ;;
        detection)      MODELS="$ALL_DETECTION_MODELS" ;;
        *) echo "[ERROR] Unknown task: $TASK"; exit 1 ;;
    esac
fi

# ── resolve wandb project ────────────────────────────────────────────────────
if [[ -z "$WANDB_PROJECT" ]]; then
    WANDB_PROJECT="marsbench-${TASK}"
fi

# ── summary ─────────────────────────────────────────────────────────────────
echo "================================================================"
echo " MarsBench Benchmark Runner"
echo "================================================================"
echo "  task          : $TASK"
echo "  data_name     : $DATA_NAME"
echo "  repo_id       : $REPO_ID"
echo "  training_type : $TRAINING_TYPE"
echo "  epochs        : $EPOCHS"
echo "  lr            : $LR"
echo "  batch_size    : $BATCH_SIZE"
echo "  num_workers   : $NUM_WORKERS"
echo "  load_from_hf  : $LOAD_FROM_HF"
echo "  wandb_project : $WANDB_PROJECT"
echo "  models        : $MODELS"
echo "================================================================"
echo ""

# ── result tracking ──────────────────────────────────────────────────────────
RESULTS_DIR="benchmark_results/${TASK}/${DATA_NAME}"
mkdir -p "$RESULTS_DIR"
SUMMARY_FILE="${RESULTS_DIR}/summary_$(date +%Y-%m-%d_%H-%M-%S).txt"

echo "Task: $TASK | Dataset: $DATA_NAME | training_type: $TRAINING_TYPE | lr: $LR" > "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "------------------------------------------------------------" >> "$SUMMARY_FILE"

TOTAL=$(echo $MODELS | wc -w)
IDX=0
FAILED_MODELS=()

# ── main loop ────────────────────────────────────────────────────────────────
for MODEL in $MODELS; do
    IDX=$((IDX + 1))
    echo ""
    echo "================================================================"
    echo " [$IDX/$TOTAL] Model: $MODEL"
    echo "================================================================"

    CMD="HF_ENDPOINT=https://hf-mirror.com python3 -m marsbench.main \
        task=${TASK} \
        model_name=${MODEL} \
        data_name=${DATA_NAME} \
        training_type=${TRAINING_TYPE} \
        training=dev \
        training.trainer.max_epochs=${EPOCHS} \
        training.optimizer.lr=${LR} \
        training.batch_size=${BATCH_SIZE} \
        training.num_workers=${NUM_WORKERS} \
        load_from_hf=${LOAD_FROM_HF} \
        logger.wandb.project=${WANDB_PROJECT}"

    if [[ -n "$REPO_ID" ]]; then
        CMD="${CMD} repo_id=\"${REPO_ID}\""
    fi

    if [[ -n "$EXTRA" ]]; then
        CMD="${CMD} ${EXTRA}"
    fi

    echo "Command: $CMD"
    echo ""

    START_TIME=$(date +%s)

    if eval "$CMD"; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        STATUS="OK"
        echo "[OK] $MODEL finished in ${ELAPSED}s"
    else
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        STATUS="FAILED"
        FAILED_MODELS+=("$MODEL")
        echo "[FAILED] $MODEL failed after ${ELAPSED}s — continuing to next model"
    fi

    echo "  ${MODEL}: ${STATUS} (${ELAPSED}s)" >> "$SUMMARY_FILE"
done

# ── final summary ────────────────────────────────────────────────────────────
echo "" >> "$SUMMARY_FILE"
echo "Finished: $(date)" >> "$SUMMARY_FILE"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo "Failed models: ${FAILED_MODELS[*]}" >> "$SUMMARY_FILE"
fi

echo ""
echo "================================================================"
echo " All done. Results summary: $SUMMARY_FILE"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo " Failed models: ${FAILED_MODELS[*]}"
fi
echo "================================================================"
