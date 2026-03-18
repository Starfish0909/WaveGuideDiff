#!/bin/bash

# ========================================
# PyIQA Image Inpainting Evaluation Script
# ========================================

# Set paths
GT_DIR=""
PRED_DIR=""

# Local weights path
WEIGHTS_DIR="$(dirname "$0")/checkpoints"


echo "=========================================="
echo "Full Image Inpainting Evaluation"
echo "Includes all full-reference metrics + FID"
echo "=========================================="
echo ""
echo "GT directory:   $GT_DIR"
echo "Prediction directory:  $PRED_DIR"
echo "Device:      cuda"
echo "Include FID:  Yes"
echo ""
echo "Starting evaluation..."
echo "=========================================="
echo ""

python pyiqa_evaluation.py \
    --gt_dir "$GT_DIR" \
    --pred_dir "$PRED_DIR" \
    --device cuda \
    --compute_fid

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="