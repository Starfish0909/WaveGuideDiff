#!/bin/bash

# ============================================================================
# WaveGuideDiff (Wavelet-Guidewire Diffusion Model) Training Script
#
# Model description:
#   WaveGuideDiff is a wavelet-guided diffusion model for IVOCT guidewire artifact removal
#
# Core modules:
#   1. DWT Encoding: DWT downsampling, separating low and high frequency components
#   2. IDWT Expanding: IDWT upsampling, frequency domain reconstruction
#   3. FSAM: Frequency-selective attention module, enhancing high-frequency details
#   4. BFIM: Bi-directional frequency interaction module
#   5. SAFM: Spatial-aware feature merging, cross-layer skip connections
#   6. MSR: Multi-scale residual connections, global information flow
#
# Usage:
#   ./train_waveguidediff.sh
# ============================================================================

export OPENAI_LOGDIR="/mnt/user/Checkpoints/WaveGuideDiff/log_cosine_batch4"
export DATA_DIR='/mnt/user/Data/Guide_Inpainting_Polar/Nowith_guide/augmented_train_images'
export OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard"

# ============================================================================
# Training config: auto-compute steps based on epochs
# ============================================================================
NUM_IMAGES=3594        # Number of images in dataset
TARGET_EPOCHS=112      # Target training epochs (adjust as needed)
BATCH_SIZE=4           # Batch size (effective batch size)
MICROBATCH=4           # Forward pass batch size (for gradient accumulation)
                       # Set to 2 if GPU memory is insufficient
                       # Gradient accumulation steps = BATCH_SIZE / MICROBATCH

# Auto-compute training steps
STEPS_PER_EPOCH=$((NUM_IMAGES / BATCH_SIZE))
TOTAL_STEPS=$((STEPS_PER_EPOCH * TARGET_EPOCHS))

echo "=========================================="
echo "Training WaveGuideDiff (Full) (Module A + B + C)"
echo "=========================================="
echo "Training config:"
echo "  - Dataset size: $NUM_IMAGES images"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Target epochs: $TARGET_EPOCHS"
echo "  - Steps per epoch: $STEPS_PER_EPOCH steps"
echo "  - Total training steps: $TOTAL_STEPS steps"
echo "=========================================="

# WaveGuideDiff model parameter config
# Explicitly specify all key parameters to ensure correct configuration
MODEL_FLAGS="--image_size 512 \
    --num_channels 128 \
    --num_res_blocks 2 \
    --learn_sigma True \
    --num_in_channels 3 \
    --num_out_channels 6 \
    --model_type waveguidediff \
    --num_heads 3,6,12,24 \
    --patch_size 4 \
    --embed_dim 96 \
    --depths 2,2,2,2 \
    --window_size 8"

# Diffusion model config
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

# Training config (using auto-computed steps)
TRAIN_FLAGS="--lr 1e-4 \
    --batch_size $BATCH_SIZE \
    --microbatch $MICROBATCH \
    --use_fp16 False \
    --fp16_scale_growth 1e-3 \
    --schedule_sampler loss-second-moment \
    --lr_anneal_steps $TOTAL_STEPS \
    --log_interval 1000 \
    --save_interval 1000 \
    --seed 42"

# Visualization config
VIZ_GT_PATH="/mnt/user/Data/Guide_Inpainting_Polar/Nowith_guide/test_images/020907(3T)_IMG0011G.png"
VIZ_MASK_PATH="/mnt/user/Data/Guide_Inpainting_Polar/Nowith_guide/test_masks/020907(3T)_IMG0011G.png"
VIZ_GT_DIR="/mnt/user/Data/Guide_Inpainting_Polar/Nowith_guide/test_images"
VIZ_MASK_DIR="/mnt/user/Data/Guide_Inpainting_Polar/Nowith_guide/test_masks"
VIZ_DIFFUSION_STEPS="1000"
VIZ_RESPACING="250"
VIZ_SCHEDULE_JUMP_PARAMS="{\\\"t_T\\\": 250, \\\"n_sample\\\": 1, \\\"jump_length\\\": 1, \\\"jump_n_sample\\\": 5}"
VIZ_OUTPUT_DIR="val_denoising_progress_with_mask_waveguidediff"
VIZ_FLAGS="--viz_gt_path ${VIZ_GT_PATH} \
    --viz_mask_path ${VIZ_MASK_PATH} \
    --viz_gt_dir ${VIZ_GT_DIR} \
    --viz_mask_dir ${VIZ_MASK_DIR} \
    --viz_diffusion_steps ${VIZ_DIFFUSION_STEPS} \
    --viz_output_dir ${VIZ_OUTPUT_DIR}"

export OUTPUT_DIR="/mnt/user/Checkpoints/WaveGuideDiff"

echo "=========================================="
echo "WaveGuideDiff Full model (Module A + B + C)"
echo "=========================================="
echo ""
echo "Module composition:"
echo "  - Module A: DWT/IDWT + FSAM (Frequency domain foundation)"
echo "  - Module B: BFIM (Bi-directional frequency interaction)"
echo "  - Module C: FSDC + MSR + SAFM (Information preservation)"
echo ""
echo "Core innovations:"
echo "  1. FSAM (Frequency-Selective Attention) - Module A"
echo "     Selective enhancement of high-frequency components"
echo ""
echo "  2. BFIM (Bi-directional Frequency Interaction) - Module B"
echo "     Bi-directional information exchange between low and high frequencies"
echo ""
echo "  3. FSDC (Frequency-aware Compression) - Module C"
echo "     Grouped compression, maintaining frequency component independence"
echo ""
echo "  4. MSR (Multi-Scale Residual) - Module C"
echo "     Global residual connections, preserving original information"
echo ""
echo "  5. SAFM (Spatial-Aware Feature Merging) - Module C"
echo "     Smart skip connection fusion"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

CHECKPOINT_PATH=""

# To resume training from checkpoint, uncomment the line below and set the path
# CHECKPOINT_PATH="/mnt/user/Checkpoints/WaveGuideDiff/model050000.pt"

if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Resuming training from checkpoint: $CHECKPOINT_PATH"
    python scripts/image_train.py \
        --data_dir $DATA_DIR \
        $MODEL_FLAGS \
        $DIFFUSION_FLAGS \
        $TRAIN_FLAGS \
        $VIZ_FLAGS \
        --viz_respacing "$VIZ_RESPACING" \
        --viz_schedule_jump_params "$VIZ_SCHEDULE_JUMP_PARAMS" \
        --resume_checkpoint $CHECKPOINT_PATH
else
    python scripts/image_train.py \
        --data_dir $DATA_DIR \
        $MODEL_FLAGS \
        $DIFFUSION_FLAGS \
        $TRAIN_FLAGS \
        $VIZ_FLAGS \
        --viz_respacing "$VIZ_RESPACING" \
        --viz_schedule_jump_params "$VIZ_SCHEDULE_JUMP_PARAMS"
fi

echo ""
echo "=========================================="
echo "Training complete"
echo "=========================================="
echo "Checkpoints saved at: $OPENAI_LOGDIR"
echo "View training logs with:"
echo "  tensorboard --logdir=$OPENAI_LOGDIR"
echo ""
echo "Inference command example:"
echo "  cd /mnt/user/WaveGuideDiff/inference"
echo "  bash inference_waveguidediff.sh"
echo "=========================================="
