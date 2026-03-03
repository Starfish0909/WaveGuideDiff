#!/usr/bin/env python3
"""
Unified inference test script
Supports skipping already-inferred images
"""

import os
import sys
import argparse
import torch as th
import torch.nn.functional as F
import time
import numpy as np
import cv2
import conf_mgt
from utils import yamlread

# Add training code path (for importing improved_diffusion)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
training_scripts_path = os.path.join(project_root, 'training', 'scripts')
if training_scripts_path not in sys.path:
    sys.path.insert(0, training_scripts_path)

# Import dist_util from inference directory (supports device parameter)
from guided_diffusion import dist_util

# Workaround for ctypes bug
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

# Import from inference directory (supports cond_fn, return_all, conf parameters)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


def toU8(sample):
    """Convert to uint8 format."""
    if sample is None:
        return sample
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
    return sample


def check_output_exists(img_name, conf, dset, eval_name):
    """Check if output file already exists."""
    srs_dir = os.path.expanduser(conf['data'][dset][eval_name]['paths']['srs'])
    srs_path = os.path.join(srs_dir, img_name)
    return os.path.exists(srs_path)


def test_model_loading():
    """Test model loading."""
    print("Testing model loading...")

    try:
        from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

        # Test model creation
        print("1. Testing model creation...")
        config = model_and_diffusion_defaults()

        # Check if model_type parameter is supported
        try:
            config['model_type'] = 'unet'
            model, diffusion = create_model_and_diffusion(**config)
            print(f"Model created successfully (model_type supported): {type(model).__name__}")
        except TypeError:
            # If model_type not supported, remove it
            config.pop('model_type', None)
            model, diffusion = create_model_and_diffusion(**config)
            print(f"Model created successfully (model_type not supported): {type(model).__name__}")

        return True

    except Exception as e:
        print(f"Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading(conf_path):
    """Test config file loading."""
    print("Testing config file loading...")

    try:
        with open(conf_path, 'r') as f:
            conf = yamlread(conf_path)
        print(f"Config loaded successfully: {conf_path}")
        print(f"   Model type: {conf.get('name', 'UNet')}")
        print(f"   Model path: {conf.get('model_path', 'N/A')}")
        return True

    except Exception as e:
        print(f"Config loading test failed: {e}")
        return False


def check_model_files(conf_path):
    """Check if model file exists."""
    print("Checking model files...")

    try:
        conf = yamlread(conf_path)
        model_path = conf.get('model_path', '')

        if os.path.exists(model_path):
            print(f"Model file found: {model_path}")
            return True
        else:
            print(f"Model file not found: {model_path}")
            return False

    except Exception as e:
        print(f"Model file check failed: {e}")
        return False


def main(conf: conf_mgt.Default_Conf, skip_existing=False, conf_path=None):
    """Main inference function."""
    print("=" * 60)
    print(f"Starting inference: {conf['name']}")
    if skip_existing:
        print("Skip existing enabled - automatically skipping already-inferred images")
    print("=" * 60)
    # Set random seed for reproducibility
    seed = conf.get("seed", None)
    if seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to: {seed}")


    device = dist_util.dev(conf.get('device'))

    # === Model and diffusion ===
    # Prepare model arguments
    model_args = args_to_dict(conf, model_and_diffusion_defaults().keys())

    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(**model_args, conf=conf)

    # === Load model weights (filter attn_mask) ===
    state_dict = dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    filtered_state_dict = {k: v for k, v in state_dict.items() if "attn_mask" not in k}
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    print("Model weights loaded (attn_mask filtered)")

    # Only warn when many keys are missing
    if missing_keys and len(missing_keys) > 10:
        print(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys and len(unexpected_keys) > 10:
        print(f"Unexpected keys: {len(unexpected_keys)}")

    # === Move model to device and set precision (after loading weights) ===
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    cond_fn = None  # No classifier guidance
    show_progress = conf.show_progress

    # Check if edge-guided model
    is_edge_model = conf.get('model_type', '') in ['swin_unet_edge', 'swin_unet_edge_guided', 'edge_fusion']
    use_edge_guidance = conf.get('edge_guidance', False) and is_edge_model
    if use_edge_guidance:
        print("Edge guidance enabled - extracting edges from GT images as structural prior")

    def model_fn(x, t, y=None, gt=None, **kwargs):
        # For edge-guided models, pass clean_image for edge extraction
        if use_edge_guidance and gt is not None:
            return model(x, t, clean_image=gt)
        return model(x, t)

    print("Starting inference...")

    dset = 'eval'
    eval_name = conf.get_default_eval_name()

    # Select data loader based on skip_existing flag
    if skip_existing:
        dl = conf.get_dataloader(dset=dset, dsName=eval_name, skip_existing=True)
    else:
        dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    # Statistics
    total_processed = 0
    total_skipped = 0

    # Show info if skip_existing is enabled
    if skip_existing:
        print("Skip existing enabled, will automatically skip already-inferred images")

    for batch in iter(dl):
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {"gt": batch["GT"]}
        gt_keep_mask = batch.get("gt_keep_mask")
        if gt_keep_mask is not None:
            model_kwargs["gt_keep_mask"] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]
        img_names = batch['GT_name']
        print(f"Inferring {batch_size} images: {img_names}")

        # Select sampling method based on config
        sampler_type = getattr(conf, 'sampler_type', 'ddpm')

        if conf.use_ddim or sampler_type == 'ddim':
            print(f"   Using DDIM sampling (eta={conf.get('ddim_eta', 0.0)})")
            result = diffusion.ddim_sample_loop(
                model_fn,
                (batch_size, 3, conf.image_size, conf.image_size),
                clip_denoised=conf.clip_denoised,
                model_kwargs=model_kwargs,
                device=device,
                progress=show_progress,
                eta=conf.get('ddim_eta', 0.0),
                conf=conf,
            )
        else:
            print(f"   Using DDPM sampling")
            result = diffusion.p_sample_loop(
                model_fn,
                (batch_size, 3, conf.image_size, conf.image_size),
                clip_denoised=conf.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                progress=show_progress,
                return_all=True,
                conf=conf,
            )

        # Save gts first (GT loaded by DataLoader, after three-step resize),
        # then composite srs in uint8 space: background from gts, inpainted region from model output.
        # This ensures srs background matches gts exactly, both from DataLoader resize.

        # Convert images to uint8
        gts = toU8(result['gt'])  # GT from DataLoader (after three-step high-quality resize)
        sample_only = toU8(result['sample'])  # Raw model output (may have background perturbation)
        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') +
                   (-1) * th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        # Composite srs in uint8 space: background=gts, inpainted region=model output
        srs = []
        for i in range(len(gts)):
            # Get mask (uint8 space)
            # gt_keep_masks[i] after toU8 is (H,W,3), values in [0,255]
            # Take first channel and binarize
            mask_uint8 = gt_keep_masks[i][:, :, 0]  # Single channel (H, W)
            mask_uint8 = (mask_uint8 > 127).astype(np.uint8) * 255  # Binarize
            mask_3ch = np.stack([mask_uint8] * 3, axis=-1)  # Expand to 3 channels (H, W, 3)

            # Composite in uint8 space (avoids floating point precision loss)
            # mask_3ch=255: background region, use gts (DataLoader GT)
            # mask_3ch=0: inpainted region, use model output
            srs_single = np.where(
                mask_3ch == 255,
                gts[i],        # Background: DataLoader GT (perfect match)
                sample_only[i] # Inpainted region: model output
            )
            srs.append(srs_single)

        srs = np.array(srs)

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=img_names, dset=dset, name=eval_name, verify_same=False
        )

        total_processed += batch_size
        print(f"Inference done, processed: {total_processed} images")

    print("=" * 60)
    print("Inference complete!")
    if skip_existing:
        print(f"Statistics:")
        print(f"   Newly inferred: {total_processed} images")
        print(f"   Skipped: {total_skipped} images")
        print(f"   Total checked: {total_processed + total_skipped} images")
    else:
        print(f"Total inferred: {total_processed} images")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified inference script')
    parser.add_argument('--conf_path', type=str, required=True, help='Config file path')
    parser.add_argument('--skip_existing', action='store_true', help='Skip already-inferred images')
    parser.add_argument('--test_only', action='store_true', help='Run tests only, no inference')
    args = parser.parse_args()

    # Load config
    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.conf_path))

    if args.test_only:
        # Run tests only
        print("Running inference tests...")

        # Test model loading
        model_test_success = test_model_loading()

        # Test config loading
        config_test_success = test_config_loading(args.conf_path)

        # Check model files
        model_exists = check_model_files(args.conf_path)

        print("\n" + "=" * 60)
        print("Test results summary:")
        print(f"Model loading: {'PASS' if model_test_success else 'FAIL'}")
        print(f"Config loading: {'PASS' if config_test_success else 'FAIL'}")
        print(f"Model file: {'EXISTS' if model_exists else 'NOT FOUND'}")

        if model_test_success and config_test_success and model_exists:
            print("\nAll tests passed! Ready for inference")
            print(f"\nUsage:")
            print(f"1. Normal inference: python test.py --conf_path {args.conf_path}")
            print(f"2. Skip existing: python test.py --conf_path {args.conf_path} --skip_existing")
            print(f"3. Use script: ./inference_waveguidediff.sh")
        else:
            print("\nSome tests failed, please check configuration")
    else:
        # Run inference
        main(conf_arg, args.skip_existing, args.conf_path)
