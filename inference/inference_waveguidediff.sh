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

echo ""
echo "==========================================="
echo "WaveGuideDiff Technical Architecture"
echo "==========================================="
echo ""
echo "Core Components:"
echo ""
echo "1. DWT Encoding (DWT Downsampling):"
echo "   Input(H,W,C) -> DWT Decomposition -> LL(low-freq) + HF(high-freq)"
echo "   |- LL component: Preserves overall structure and semantic info"
echo "   |- HF component: LH, HL, HH edge textures in three directions"
echo "   |- FSAM: Frequency-selective attention, adaptively enhancing important high-freq"
echo "   |- BFIM: Bi-directional interaction, LL and HF mutually enhance"
echo "   +- FSDC: Frequency-aware compression, parameter-efficient dim transform"
echo ""
echo "2. IDWT Expanding (IDWT Upsampling):"
echo "   Input(H,W,C) -> Linear expand 4x channels -> IDWT -> (2H,2W,C/2)"
echo "   |- Learned allocation: Network learns how to distribute to 4 channels"
echo "   |- Frequency domain reconstruction: IDWT ensures frequency info consistency"
echo "   +- No artifacts: Avoids checkerboard artifacts from traditional upsampling"
echo ""
echo "3. FSAM (Frequency-Selective Attention Module):"
echo "   [LH, HL, HH] -> Stack -> Channel Attention -> Weighted fusion"
echo "   |- Adaptive weights: Learns importance of different directional high-freq"
echo "   |- Global modeling: Global Pooling + MLP"
echo "   +- Detail enhancement: Highlights key edges and textures"
echo ""
echo "4. BFIM (Bi-directional Frequency Interaction Module):"
echo "   LL <-> HF Bi-directional interaction"
echo "   |- LL -> HF: Structure guides details"
echo "   |- HF -> LL: Details complement structure"
echo "   +- Parameter efficient: HF compressed before interaction"
echo ""
echo "5. SAFM (Spatial-Aware Feature Merging):"
echo "   [Decoder features, Encoder features] -> Channel Attention -> Fusion"
echo "   |- Smart skip: More efficient than simple concat"
echo "   |- Adaptive fusion: Learns optimal fusion strategy"
echo "   +- Reduces redundancy: Reduces feature repetition"
echo ""
echo "6. MSR (Multi-Scale Residual):"
echo "   Input -> [Conv branch, Identity branch] -> Add"
echo "   |- Global residual: Enhances gradient flow"
echo "   |- Multi-scale: Preserves information at different levels"
echo "   +- Stabilizes training: Avoids vanishing gradients"
echo ""
echo "Inference parameters:"
echo "  - Sampling method: DDPM (jump sampling)"
echo "  - Sampling steps: 250 steps"
echo "  - Jump parameters: t_T=250, jump_length=1, jump_n_sample=5"
echo "  - Random seed: 36"
echo ""
echo ""
echo "==========================================="
echo ""
echo "Starting inference (auto-skipping already-inferred images)..."
echo ""

export PYTHONWARNINGS="ignore::FutureWarning"
export CUDA_LAUNCH_BLOCKING=0

START_TIME=$(date +%s)

python test.py --conf_path $CONF_FILE --skip_existing

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "==========================================="
echo "WaveGuideDiff Inference complete!"
echo "==========================================="
echo "Inference time: ${MINUTES}min ${SECONDS}sec"
echo "Output: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  - Inpainted images: $OUTPUT_DIR/srs/"
echo "  - Input images: $OUTPUT_DIR/lrs/"
echo "  - Ground truth images: $OUTPUT_DIR/gts/"
echo "  - Mask images: $OUTPUT_DIR/gt_keep_masks/"
echo "==========================================="
