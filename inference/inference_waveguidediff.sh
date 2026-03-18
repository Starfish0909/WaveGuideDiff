#!/bin/bash

# ============================================================================
# WaveGuideDiff (Wavelet-Guidewire Diffusion Model) Inference Script
# ============================================================================

echo "==========================================="
echo "WaveGuideDiff
echo "==========================================="

CONF_FILE="confs/waveguidediff.yml"
if [ ! -f "$CONF_FILE" ]; then
    echo "Config file not found: $CONF_FILE"
    exit 1
fi

MODEL_PATH=$(grep "model_path:" $CONF_FILE | sed "s/.*model_path: *['\"]*//" | sed "s/['\"].*//")
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file not found: $MODEL_PATH"
    echo "Please ensure the model has been trained"
    echo "Or update model_path in config to point to the correct checkpoint"
    exit 1
fi

echo "Config file: $CONF_FILE"
echo "Model path: $MODEL_PATH"
echo "Model type: waveguidediff"

OUTPUT_DIR=$(grep "srs:" $CONF_FILE | sed "s/.*srs: *['\"]*//" | sed "s/['\"].*//" | cut -d'/' -f1-7)
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"


export PYTHONWARNINGS="ignore::FutureWarning"
export CUDA_LAUNCH_BLOCKING=0

START_TIME=$(date +%s)

python test.py --conf_path $CONF_FILE --skip_existing

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

