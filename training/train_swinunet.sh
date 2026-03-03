#!/bin/bash

export OPENAI_LOGDIR="/mnt/user/Checkpoints/WaveGuideDiff_swinunet_baseline/log_cosine_batch4"
export DATA_DIR='/mnt/user/Data/Guide_Inpainting_Polar/Nowith_guide/augmented_train_images'
export OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard"

# ============================================================================
# Training config: auto-compute steps based on epochs
# ============================================================================
NUM_IMAGES=3594        # Number of images in dataset
TARGET_EPOCHS=112      # Target training epochs
BATCH_SIZE=4          # Batch size (effective batch size)
MICROBATCH=4     # Forward pass batch size (for gradient accumulation)
                       # Set to 8 or 12 if GPU memory is insufficient
                       # Gradient accumulation steps = BATCH_SIZE / MICROBATCH

# Auto-compute training steps
STEPS_PER_EPOCH=$((NUM_IMAGES / BATCH_SIZE))
TOTAL_STEPS=$((STEPS_PER_EPOCH * TARGET_EPOCHS))

# SwinUNet model parameter config
MODEL_FLAGS="--image_size 512 --num_channels 128 --num_res_blocks 3 --learn_sigma True --num_in_channels 3 --num_out_channels 6 --model_type swin_unet"

# No SNR weighting or perceptual loss - Baseline config
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

TRAIN_FLAGS="--lr 1e-4 --batch_size $BATCH_SIZE --microbatch $MICROBATCH --use_fp16 False --fp16_scale_growth 1e-3 --schedule_sampler loss-second-moment --lr_anneal_steps $TOTAL_STEPS --log_interval 1000 --save_interval 1000 --seed 42"

# Visualization config
VIZ_GT_PATH="/mnt/user/Data/Guide_Inpainting_Polar/Nowith_guide/test_images/020907(3T)_IMG0011G.png"
VIZ_MASK_PATH="/mnt/user/Data/Guide_Inpainting_Polar/Nowith_guide/test_masks/020907(3T)_IMG0011G.png"
VIZ_GT_DIR="/mnt/user/Data/Guide_Inpainting_Polar/Nowith_guide/test_images"
VIZ_MASK_DIR="/mnt/user/Data/Guide_Inpainting_Polar/Nowith_guide/test_masks"
VIZ_DIFFUSION_STEPS="1000"
VIZ_RESPACING="250"
VIZ_SCHEDULE_JUMP_PARAMS="{\"t_T\": 250, \"n_sample\": 1, \"jump_length\": 1, \"jump_n_sample\": 5}"
VIZ_OUTPUT_DIR="val_denoising_progress_with_mask_baseline"
VIZ_FLAGS="--viz_gt_path ${VIZ_GT_PATH} --viz_mask_path ${VIZ_MASK_PATH} --viz_gt_dir ${VIZ_GT_DIR} --viz_mask_dir ${VIZ_MASK_DIR} --viz_diffusion_steps ${VIZ_DIFFUSION_STEPS} --viz_output_dir ${VIZ_OUTPUT_DIR}"

export OUTPUT_DIR="/mnt/user/Checkpoints/WaveGuideDiff_swinunet_baseline"

# =============================================================================
# Auto-detect config parameters
# =============================================================================
# Extract parameters from TRAIN_FLAGS
BATCH_SIZE=$(echo "$TRAIN_FLAGS" | grep -oP '(?<=--batch_size )\d+' || echo "not specified")
LEARNING_RATE=$(echo "$TRAIN_FLAGS" | grep -oP '(?<=--lr )[^\s]+' || echo "not specified")
USE_FP16=$(echo "$TRAIN_FLAGS" | grep -oP '(?<=--use_fp16 )[^\s]+' || echo "False")

# Extract parameters from DIFFUSION_FLAGS
DIFFUSION_STEPS=$(echo "$DIFFUSION_FLAGS" | grep -oP '(?<=--diffusion_steps )\d+' || echo "not specified")
NOISE_SCHEDULE=$(echo "$DIFFUSION_FLAGS" | grep -oP '(?<=--noise_schedule )[^\s]+' || echo "not specified")

# Extract parameters from MODEL_FLAGS
IMAGE_SIZE=$(echo "$MODEL_FLAGS" | grep -oP '(?<=--image_size )\d+' || echo "not specified")
NUM_IN_CHANNELS=$(echo "$MODEL_FLAGS" | grep -oP '(?<=--num_in_channels )\d+' || echo "not specified")
NUM_OUT_CHANNELS=$(echo "$MODEL_FLAGS" | grep -oP '(?<=--num_out_channels )\d+' || echo "not specified")
MODEL_TYPE=$(echo "$MODEL_FLAGS" | grep -oP '(?<=--model_type )[^\s]+' || echo "not specified")

# Check mixed precision training status
if [[ "$USE_FP16" == "True" || "$USE_FP16" == "true" ]]; then
    FP16_STATUS="Mixed precision (FP16): enabled (use_fp16=True)"
else
    FP16_STATUS="Mixed precision (FP16): disabled (use_fp16=False)"
fi

# Check if SNR weighting is enabled
if echo "$DIFFUSION_FLAGS" | grep -q "use_snr_weight"; then
    SNR_STATUS="SNR weighting: enabled"
else
    SNR_STATUS="SNR weighting: disabled"
fi

# Check if perceptual loss is enabled
if echo "$DIFFUSION_FLAGS" | grep -q "perceptual_weight"; then
    PERCEPTUAL_WEIGHT=$(echo "$DIFFUSION_FLAGS" | grep -oP '(?<=--perceptual_weight )[^\s]+' || echo "0.0")
    if [[ "$PERCEPTUAL_WEIGHT" != "0.0" && "$PERCEPTUAL_WEIGHT" != "0" ]]; then
        PERCEPTUAL_STATUS="Perceptual loss: enabled (weight=${PERCEPTUAL_WEIGHT})"
    else
        PERCEPTUAL_STATUS="Perceptual loss: disabled"
    fi
else
    PERCEPTUAL_STATUS="Perceptual loss: disabled"
fi


echo "=========================================="
echo "Training Baseline model (pure MSE loss)"
echo "=========================================="
echo "  ${SNR_STATUS}"
echo "  ${PERCEPTUAL_STATUS}"
echo "  Loss function: pure MSE"
echo "  Output directory: $OUTPUT_DIR"
echo "=========================================="

CHECKPOINT_PATH=""

python scripts/image_train.py \
    --data_dir $DATA_DIR \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS \
    $VIZ_FLAGS \
    --viz_respacing "$VIZ_RESPACING" \
    --viz_schedule_jump_params "$VIZ_SCHEDULE_JUMP_PARAMS"

echo "Baseline model training complete"
