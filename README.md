# WaveGuideDiff

Official PyTorch implementation of **WaveGuideDiff: A Wavelet-Guided Diffusion Model for Guidewire Artifact Removal in IVOCT Images**.

## Overview

WaveGuideDiff is a wavelet-guided diffusion model designed for removing guidewire artifacts from Intravascular Optical Coherence Tomography (IVOCT) images. The model leverages wavelet transforms to effectively separate and process frequency components, enabling high-quality artifact removal while preserving important structural details.

### Key Features

- **Wavelet-Guided Architecture**: Utilizes Discrete Wavelet Transform (DWT) and Inverse DWT (IDWT) for frequency-aware processing
- **Modular Design**: Incremental ablation study with three core modules (A, B, C)
- **Frequency-Selective Attention (FSAM)**: Enhances high-frequency details critical for medical imaging
- **Bi-directional Frequency Interaction (BFIM)**: Enables information exchange between frequency bands
- **Multi-Scale Residual (MSR)**: Preserves global information flow across the network
- **Spatial-Aware Feature Merging (SAFM)**: Intelligent skip connection fusion

## Architecture

WaveGuideDiff consists of three incremental modules:

- **Module A**: DWT/IDWT + FSAM (Frequency Domain Foundation)
- **Module B**: BFIM (Bi-directional Frequency Interaction)
- **Module C**: FSDC + MSR + SAFM (Information Preservation)

### Core Components

1. **DWT Encoding**: Separates input into low and high frequency components
2. **FSAM**: Frequency-selective attention for high-frequency enhancement
3. **BFIM**: Bi-directional information exchange between frequency bands
4. **FSDC**: Frequency-aware spatial dimension compression
5. **MSR**: Multi-scale residual connections for global information flow
6. **SAFM**: Spatial-aware feature merging for skip connections
7. **IDWT Expanding**: Reconstructs output from frequency components

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.3 (for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Starfish0909/WaveGuideDiff.git
cd WaveGuideDiff
```

2. Install dependencies:
```bash
# Install training dependencies
cd training
pip install -e .

# Install evaluation dependencies
pip install pyiqa opencv-python tqdm
```

## Project Structure

```
WaveGuideDiff/
├── training/                    # Training code
│   ├── scripts/
│   │   ├── image_train.py      # Main training script
│   │   └── improved_diffusion/
│   │       ├── waveguidediff.py    # WaveGuideDiff model (Full)
│   │       ├── swin_unet.py        # SwinUNet baseline
│   │       └── ...
│   ├── train_waveguidediff.sh  # Training script for WaveGuideDiff
│   └── train_swinunet.sh       # Training script for SwinUNet
├── inference/                   # Inference code
│   ├── test.py                 # Main inference script
│   ├── confs/
│   │   ├── waveguidediff.yml   # WaveGuideDiff config
│   │   └── swin_unet.yml       # SwinUNet config
│   └── inference_waveguidediff.sh
├── evaluation/                  # Evaluation code
│   ├── pyiqa_evaluation.py     # PyIQA-based evaluation
│   └── run_pyiqa_example.sh
└── shared/
    └── model_registry.py       # Shared model registry
```

## Usage

### Training

1. Prepare your dataset in the following structure:
```
data/
├── train_images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── train_masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

2. Update paths in `training/train_waveguidediff.sh`:
```bash
export DATA_DIR='/path/to/your/train_images'
export OPENAI_LOGDIR="/path/to/save/checkpoints"
```

3. Start training:
```bash
cd training
bash train_waveguidediff.sh
```

### Inference

1. Update the model path and data paths in `inference/confs/waveguidediff.yml`:
```yaml
model_path: '/path/to/checkpoint.pt'
data:
  eval:
    paper_face_mask:
      gt_path: "/path/to/test_images"
      mask_path: "/path/to/test_masks"
```

2. Run inference:
```bash
cd inference
python test.py --conf_path confs/waveguidediff.yml
```

Or use the provided script:
```bash
cd inference
bash inference_waveguidediff.sh
```

### Evaluation

Evaluate results using PyIQA metrics (PSNR, SSIM, LPIPS, DISTS, FID):

```bash
cd evaluation
python pyiqa_evaluation.py \
    --gt_dir /path/to/ground_truth \
    --pred_dir /path/to/predictions \
    --device cuda \
    --compute_fid
```


## Supported Models

The codebase supports multiple model architectures through a unified registry system:

- **WaveGuideDiff**: Full model with all modules (A+B+C)
- **SwinUNet**: Swin Transformer-based UNet baseline

To add a new model, simply register it in `shared/model_registry.py`.

## Evaluation Metrics

The evaluation script supports the following metrics:

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **SSIM** (Structural Similarity Index): Higher is better
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better
- **DISTS** (Deep Image Structure and Texture Similarity): Lower is better
- **FID** (Fréchet Inception Distance): Lower is better

## Citation

If you find this work useful, please cite:

```bibtex
@article{waveguidediff2026,
  title={WaveGuideDiff: A Wavelet-Guided Diffusion Model for Guidewire Artifact Removal in IVOCT Images},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](training/LICENSE) file for details.

## Acknowledgments

This codebase is built upon:
- [OpenAI's Improved Diffusion](https://github.com/openai/improved-diffusion)
- [PyIQA](https://github.com/chaofengc/IQA-PyTorch) for image quality assessment

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
