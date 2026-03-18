# WaveGuideDiff

Official PyTorch implementation of **WaveGuideDiff: A Wavelet-Guided Diffusion Model for Guidewire Artifact Removal in IVOCT Images**.

WaveGuideDiff is designed to repair guidewire artifacts in intravascular optical coherence tomography (IVOCT) images while preserving vessel structures and tissue details. This repository includes training, inference, and image-quality evaluation code for the WaveGuideDiff pipeline.

## Example Results

The following examples show guidewire-corrupted Cartesian images and the corresponding repaired outputs.

| Original | Repaired |
| --- | --- |
| <img src="datasets/example/cartesian_image/image1.png" width="320" alt="original image 1"> | <img src="datasets/example/cartesian_repaired/image1.png" width="320" alt="repaired image 1"> |
| <img src="datasets/example/cartesian_image/image2.png" width="320" alt="original image 2"> | <img src="datasets/example/cartesian_repaired/image2.png" width="320" alt="repaired image 2"> |
| <img src="datasets/example/cartesian_image/image3.png" width="320" alt="original image 3"> | <img src="datasets/example/cartesian_repaired/image3.png" width="320" alt="repaired image 3"> |
| <img src="datasets/example/cartesian_image/image4.png" width="320" alt="original image 4"> | <img src="datasets/example/cartesian_repaired/image4.png" width="320" alt="repaired image 4"> |

## Installation

### Environment

- Python >= 3.8
- PyTorch >= 1.12
- CUDA-compatible GPU is recommended for training and inference

### Dependency Note

This repository currently does **not** provide a top-level `requirements.txt`. Training dependencies are declared in `training/setup.py`, while some inference and evaluation packages are installed separately.

### Setup

```bash
git clone https://github.com/Starfish0909/WaveGuideDiff.git
cd WaveGuideDiff

# Install a PyTorch build that matches your CUDA environment first.
# Then install the project package and extra tools.
pip install -e ./training
pip install pyiqa opencv-python pyyaml
```

## Project Structure

```text
WaveGuideDiff/
├── training/                    # Training code
│   ├── scripts/                # Diffusion model and training utilities
│   └── train_waveguidediff.sh  # Training entry script
├── inference/                  # Inference code and configs
│   ├── confs/
│   ├── test.py
│   └── inference_waveguidediff.sh
├── evaluation/                 # Quality evaluation scripts
├── datasets/example/           # Example original and repaired images
├── model.pdf                   # Model figure
└── result.pdf                  # Example result figure
```

## Data Preparation

Prepare your training data in paired form. A typical structure is:

```text
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

Update the data paths in `training/train_waveguidediff.sh` before launching training.

## Training

Edit the relevant paths in `training/train_waveguidediff.sh`:

```bash
export DATA_DIR='/path/to/your/train_images'
export OPENAI_LOGDIR='/path/to/save/checkpoints'
```

Run training:

```bash
cd training
bash train_waveguidediff.sh
```

## Inference

Update `inference/confs/waveguidediff.yml` with your checkpoint and dataset paths:

```yaml
model_path: '/path/to/checkpoint.pt'
data:
  eval:
    paper_face_mask:
      gt_path: '/path/to/test_images'
      mask_path: '/path/to/test_masks'
```

Run inference with:

```bash
cd inference
python test.py --conf_path confs/waveguidediff.yml
```

Or use the helper script:

```bash
cd inference
bash inference_waveguidediff.sh
```

## Evaluation

Evaluate repaired results against ground truth images:

```bash
cd evaluation
python pyiqa_evaluation.py \
    --gt_dir /path/to/ground_truth \
    --pred_dir /path/to/predictions \
    --device cuda \
    --compute_fid
```

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

This project is licensed under the MIT License. See `training/LICENSE` for details.

## Acknowledgments

- [OpenAI Improved Diffusion](https://github.com/openai/improved-diffusion)
- [PyIQA](https://github.com/chaofengc/IQA-PyTorch)
