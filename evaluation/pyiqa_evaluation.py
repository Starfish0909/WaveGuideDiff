"""
Image Inpainting Quality Evaluation - Using pyiqa library
Designed specifically for Image Inpainting tasks

Supported Metrics:
Full-Reference Metrics:
- PSNR (Peak Signal-to-Noise Ratio): Higher is better
- SSIM (Structural Similarity Index): Higher is better
- LPIPS (Learned Perceptual Image Patch Similarity): Lower is better
- DISTS (Deep Image Structure and Texture Similarity): Lower is better
- FID (Fréchet Inception Distance): Lower is better
"""

import os
import torch
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import json
from datetime import datetime
import argparse


class PyIQAEvaluator:
    """PyIQA-based image quality evaluator"""

    def __init__(self, device='cuda', lpips_net='vgg', compute_fid=False,
                 local_weights_path=None):
        """
        Initialize evaluator

        Args:
            device: 'cuda' or 'cpu'
            lpips_net: LPIPS network backbone ('vgg', 'alex', 'squeeze')
                      - 'vgg': VGG network (default, recommended, most commonly used)
                      - 'alex': AlexNet (faster but slightly lower accuracy)
                      - 'squeeze': SqueezeNet (fastest but lowest accuracy)
            compute_fid: Whether to compute FID (Fréchet Inception Distance)
                        Note: FID evaluates distribution difference across the dataset, not pairwise
            local_weights_path: Local weights path, if None will auto-download
        """
        try:
            import pyiqa
        except ImportError:
            raise ImportError(
                "Please install pyiqa:\n"
                "pip install pyiqa\n"
                "or: pip install git+https://github.com/chaofengc/IQA-PyTorch.git"
            )

        self.device = device
        self.lpips_net = lpips_net
        self.compute_fid = compute_fid
        self.local_weights_path = local_weights_path

        print(f"Initializing evaluation metrics (device={device})...")

        # Setup local weights path
        if local_weights_path:
            print(f"Using local pretrained weights: {local_weights_path}")

            import os
            import shutil

            # pyiqa cache directory path
            pyiqa_cache = os.path.expanduser('~/.cache/torch/hub/pyiqa')

            # Actual weights storage directory (visible in working path)
            weights_store = os.path.join(local_weights_path, 'pyiqa_weights')
            os.makedirs(weights_store, exist_ok=True)

            # If cache directory doesn't exist or is not a symlink, create symlink
            if not os.path.exists(pyiqa_cache):
                # Create parent directory
                os.makedirs(os.path.dirname(pyiqa_cache), exist_ok=True)
                # Create symlink: ~/.cache/torch/hub/pyiqa -> /mnt/user/.../pyiqa_weights
                os.symlink(weights_store, pyiqa_cache)
                print(f"  Created weights directory symlink: {pyiqa_cache} -> {weights_store}")
            elif not os.path.islink(pyiqa_cache):
                # If it's a regular directory, backup then create link
                backup_dir = pyiqa_cache + '_backup'
                print(f"  Backing up existing cache directory: {pyiqa_cache} -> {backup_dir}")
                shutil.move(pyiqa_cache, backup_dir)
                os.symlink(weights_store, pyiqa_cache)
                print(f"  Created weights directory symlink: {pyiqa_cache} -> {weights_store}")

            # Copy/link weight files to actual storage directory
            lpips_weights = {
                'vgg': 'LPIPS_v0.1_vgg-a78928a0.pth',
                'alex': 'LPIPS_v0.1_alex-df73285e.pth',
                'squeeze': 'LPIPS_v0.1_squeeze-33a58a82.pth'
            }

            # Link LPIPS weights
            for net, weight_name in lpips_weights.items():
                src = os.path.join(local_weights_path, 'lpips', f'{net}.pth')
                dst = os.path.join(weights_store, weight_name)
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                        print(f"  Linked LPIPS-{net.upper()} weights")
                    except Exception as e:
                        print(f"  Warning: Failed to link {net} weights: {e}")

            # Link DISTS weights
            dists_src = os.path.join(local_weights_path, 'hub', 'DISTS_weights-f5e65c96.pth')
            dists_dst = os.path.join(weights_store, 'DISTS_weights-f5e65c96.pth')
            if os.path.exists(dists_src) and not os.path.exists(dists_dst):
                try:
                    os.symlink(dists_src, dists_dst)
                    print(f"  Linked DISTS weights")
                except Exception as e:
                    print(f"  Warning: Failed to link DISTS weights: {e}")

            # Link FID weights
            fid_src = os.path.join(local_weights_path, 'FID', 'pt_inception-2015-12-05-6726825d.pth')
            fid_dst = os.path.join(weights_store, 'pt_inception-2015-12-05-6726825d.pth')
            if os.path.exists(fid_src) and not os.path.exists(fid_dst):
                try:
                    os.symlink(fid_src, fid_dst)
                    print(f"  Linked FID Inception weights")
                except Exception as e:
                    print(f"  Warning: Failed to link FID weights: {e}")

            # Link PIEAPP weights (removed - not needed)

            print(f"  Weights stored at: {weights_store}")
        else:
            print(f"Note: Using online download mode")
            print(f"      Will auto-download to ~/.cache/torch/hub/pyiqa/ on first run")
        print()

        # Full-reference metrics (suitable for image inpainting tasks)
        self.fr_metrics = {}

        # Basic metrics
        print("  Loading basic metrics: PSNR, SSIM...")
        self.fr_metrics['psnr'] = pyiqa.create_metric('psnr', device=device)
        self.fr_metrics['ssim'] = pyiqa.create_metric('ssim', device=device)

        # Perceptual metrics (require pretrained weights)
        print(f"  Loading perceptual metrics: LPIPS-{lpips_net.upper()}, DISTS...")
        self.fr_metrics['lpips'] = pyiqa.create_metric('lpips', device=device, as_loss=False)
        self.fr_metrics['dists'] = pyiqa.create_metric('dists', device=device)

        # FID metric (optional, evaluates dataset distribution)
        self.fid_metric = None
        if compute_fid:
            print("  Loading FID metric (Fréchet Inception Distance)...")
            try:
                self.fid_metric = pyiqa.create_metric('fid', device=device)
                print("    Note: FID evaluates distribution difference across the dataset, not pairwise")
                print("    Using single-process mode to avoid shared memory issues")
            except Exception as e:
                print(f"    Warning: FID failed to load: {e}")

        print("All metrics loaded!\n")

    def load_image(self, img_path):
        """
        Load image and convert to tensor

        Args:
            img_path: Image path

        Returns:
            torch.Tensor: (1, C, H, W), range [0, 1]
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return img.to(self.device)

    def _print_partial_results(self, results):
        """Print partial evaluation results (internal helper function)"""
        # Basic metrics
        print("\n[Basic Metrics]")
        print("-" * 80)
        metrics_basic = [
            ('PSNR', 'psnr', '↑', 'dB'),
            ('SSIM', 'ssim', '↑', ''),
        ]
        for display_name, key, direction, unit in metrics_basic:
            if key in results and results[key] is not None:
                unit_str = f" {unit}" if unit else ""
                print(f"  {display_name:12s} {direction}:  {results[key]:8.4f}{unit_str}")

        # Perceptual metrics
        print("\n[Perceptual Metrics]")
        print("-" * 80)
        metrics_perceptual = [
            ('LPIPS', 'lpips', '↓', ''),
            ('DISTS', 'dists', '↓', ''),
        ]
        for display_name, key, direction, unit in metrics_perceptual:
            if key in results and results[key] is not None:
                print(f"  {display_name:12s} {direction}:  {results[key]:8.4f}")

        # FID metric
        if 'fid' in results and results['fid'] is not None:
            print("\n[Distribution Metrics]")
            print("-" * 80)
            print(f"  FID          ↓:  {results['fid']:8.4f}  (lower is better)")

    def compute_single_pair(self, gt_img, pred_img):
        """
        Compute all metrics for a single image pair

        Args:
            gt_img: Ground truth tensor (1, C, H, W)
            pred_img: Predicted image tensor (1, C, H, W)

        Returns:
            dict: Results of all metrics
        """
        results = {}

        # Full-reference metrics
        for name, metric in self.fr_metrics.items():
            try:
                with torch.no_grad():
                    score = metric(gt_img, pred_img)
                    results[name] = score.item() if torch.is_tensor(score) else score
            except Exception as e:
                print(f"  Warning: {name} computation failed: {e}")
                results[name] = None

        return results

    def compute_fid_score(self, gt_dir, pred_dir):
        """
        Compute FID score (Frechet Inception Distance)

        FID evaluates distribution difference between two image sets, not pairwise

        Args:
            gt_dir: Ground truth image directory
            pred_dir: Inpainted image directory

        Returns:
            float: FID score, lower is better (0 means identical distributions)
        """
        if self.fid_metric is None:
            return None

        try:
            print("\nComputing FID score...")
            print("  Note: FID needs to extract features from all images, may take a while...")
            print("  Using single-process mode (num_workers=0) to avoid shared memory issues...")

            # pyiqa's FID implementation supports passing directory paths directly
            # Pass num_workers=0 to avoid DataLoader shared memory issues
            fid_score = self.fid_metric(gt_dir, pred_dir, num_workers=0)

            if torch.is_tensor(fid_score):
                fid_score = fid_score.item()

            print(f"  FID score: {fid_score:.4f}")
            return float(fid_score)

        except Exception as e:
            print(f"  FID computation failed: {e}")
            return None

    def evaluate_dataset(self, gt_dir, pred_dir, save_json=None, max_images=None):
        """
        Evaluate entire dataset (compute and display in stages)

        Args:
            gt_dir: Ground truth image directory
            pred_dir: Inpainted image directory
            save_json: JSON file path to save results (optional)
            max_images: Limit number of images to process (optional)

        Returns:
            dict: Average metric results
        """
        # Get image list
        img_exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        gt_paths = []
        for ext in img_exts:
            gt_paths.extend(glob(os.path.join(gt_dir, ext)))
            gt_paths.extend(glob(os.path.join(gt_dir, ext.upper())))
        gt_paths = sorted(gt_paths)

        if len(gt_paths) == 0:
            raise ValueError(f"No image files found in {gt_dir}")

        # Limit number of images
        if max_images is not None and max_images > 0:
            gt_paths = gt_paths[:max_images]
            print(f"Found {len(gt_paths)} images (limited to first {max_images})\n")
        else:
            print(f"Found {len(gt_paths)} images\n")

        # Initialize result accumulators
        all_results = {}
        avg_results = {}

        # Define metric groups (ordered by actual speed, fast to slow)
        metric_groups = [
            ('Basic metrics', ['psnr', 'ssim'], ''),           # Very fast
            ('Perceptual metrics', ['lpips', 'dists'], ''),    # Medium speed
        ]

        # Compute by metric group
        for group_name, metrics, emoji in metric_groups:
            print(f"\n{'='*80}")
            print(f"Computing {group_name}: {', '.join([m.upper() for m in metrics])}")
            print('='*80)

            for metric_name in metrics:
                if metric_name not in self.fr_metrics:
                    continue

                print(f"\n[{metric_name.upper()}] Computing...")
                metric_results = []

                pbar = tqdm(gt_paths, desc=f"  {metric_name.upper():8s}", unit="img",
                           bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

                for gt_path in pbar:
                    gt_name = os.path.basename(gt_path)
                    pred_path = os.path.join(pred_dir, gt_name)

                    if not os.path.exists(pred_path):
                        continue

                    try:
                        # Load images
                        gt_img = self.load_image(gt_path)
                        pred_img = self.load_image(pred_path)

                        if gt_img.shape != pred_img.shape:
                            pred_img = torch.nn.functional.interpolate(
                                pred_img, size=(gt_img.shape[2], gt_img.shape[3]),
                                mode='bilinear', align_corners=False
                            )

                        # Compute current metric
                        with torch.no_grad():
                            score = self.fr_metrics[metric_name](gt_img, pred_img)
                            value = score.item() if torch.is_tensor(score) else score
                            metric_results.append(value)

                        # Update progress bar with current value
                        if len(metric_results) > 0:
                            pbar.set_postfix({metric_name.upper(): f"{metric_results[-1]:.4f}"})

                    except Exception as e:
                        continue

                # Save results for this metric
                all_results[metric_name] = metric_results

                # Compute average and display immediately
                if len(metric_results) > 0:
                    avg_value = float(np.mean(metric_results))
                    avg_results[metric_name] = avg_value
                    direction = '↑' if metric_name in ['psnr', 'ssim'] else '↓'
                    print(f"  {metric_name.upper():8s} {direction}: {avg_value:8.4f}  (done {len(metric_results)}/{len(gt_paths)} images)")

            # Print group summary after each group completes
            print(f"\n{'='*80}")
            print(f"{group_name} done!")
            print('='*80)
            self._print_partial_results(avg_results)
            print('='*80)

        # ========== FID: Compute dataset distribution metric (if enabled) ==========
        if self.compute_fid and self.fid_metric is not None:
            print(f"\n{'='*80}")
            print("Computing Distribution metrics: FID")
            print('='*80)
            fid_score = self.compute_fid_score(gt_dir, pred_dir)
            if fid_score is not None:
                avg_results['fid'] = fid_score
                print(f"  FID      ↓: {fid_score:8.4f}  (dataset-level)")

                # Print results after FID computation
                print(f"\n{'='*80}")
                print("Distribution metrics done!")
                print('='*80)
                self._print_partial_results(avg_results)
                print('='*80 + "\n")
        else:
            print(f"\n{'='*80}")
            print("Skipping FID computation (--compute_fid not enabled)")
            print('='*80 + "\n")

        # Save results (always save by default)
        if save_json:
            # Extract model info from path
            pred_parts = os.path.normpath(pred_dir).split(os.sep)

            # Extract model config name (second to last directory)
            model_config = pred_parts[-2] if len(pred_parts) >= 2 else "unknown"

            # Extract model series name (third to last directory)
            model_series = pred_parts[-3] if len(pred_parts) >= 3 else "unknown"

            # Define metric metadata (direction and category)
            metric_info = {
                'psnr': {'direction': '↑', 'category': 'Basic metrics', 'full_name': 'Peak Signal-to-Noise Ratio'},
                'ssim': {'direction': '↑', 'category': 'Basic metrics', 'full_name': 'Structural Similarity Index'},
                'lpips': {'direction': '↓', 'category': 'Perceptual metrics', 'full_name': 'Learned Perceptual Image Patch Similarity'},
                'dists': {'direction': '↓', 'category': 'Perceptual metrics', 'full_name': 'Deep Image Structure and Texture Similarity'},
                'fid': {'direction': '↓', 'category': 'Distribution metrics', 'full_name': 'Fréchet Inception Distance'},
            }

            # Format metric results (keep 4 decimal places)
            formatted_metrics = {}
            for key, value in avg_results.items():
                if value is not None:
                    formatted_metrics[key] = {
                        'value': round(value, 4),
                        'direction': metric_info.get(key, {}).get('direction', ''),
                        'category': metric_info.get(key, {}).get('category', 'Other'),
                        'full_name': metric_info.get(key, {}).get('full_name', key.upper())
                    }
                else:
                    formatted_metrics[key] = None

            result_data = {
                'model_series': model_series,  # e.g.: ArtiFusion_Wavelet_V3
                'model_config': model_config,  # e.g.: cosine_ddpm_250_1_1_5_100000_36_batch1
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'gt_dir': gt_dir,
                'pred_dir': pred_dir,
                'num_images': len(gt_paths),
                'metrics': formatted_metrics,
                'metrics_raw': avg_results  # Keep raw unformatted values for later processing
            }
            try:
                with open(save_json, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved to: {save_json}")
                print(f"   Model series: {model_series}")
                print(f"   Model config: {model_config}")
            except Exception as e:
                print(f"\nFailed to save results: {e}")

        return avg_results


def print_results(results):
    """Format and print evaluation results"""
    print("\n" + "=" * 80)
    print("Image Inpainting Quality Evaluation Results (based on pyiqa)")
    print("=" * 80)

    # Basic metrics
    print("\n[Basic Metrics]")
    print("-" * 80)
    metrics_basic = [
        ('PSNR', 'psnr', '↑', 'dB'),
        ('SSIM', 'ssim', '↑', ''),
    ]
    for display_name, key, direction, unit in metrics_basic:
        if key in results and results[key] is not None:
            unit_str = f" {unit}" if unit else ""
            print(f"  {display_name:12s} {direction}:  {results[key]:8.4f}{unit_str}")

    # Perceptual metrics
    print("\n[Perceptual Metrics]")
    print("-" * 80)
    metrics_perceptual = [
        ('LPIPS', 'lpips', '↓', ''),
        ('DISTS', 'dists', '↓', ''),
    ]
    for display_name, key, direction, unit in metrics_perceptual:
        if key in results and results[key] is not None:
            print(f"  {display_name:12s} {direction}:  {results[key]:8.4f}")

    # FID metric
    if 'fid' in results and results['fid'] is not None:
        print("\n[Distribution Metrics] (evaluate dataset-level distribution difference)")
        print("-" * 80)
        print(f"  FID          ↓:  {results['fid']:8.4f}  (lower is better, 0 means identical)")

    print("\n" + "=" * 80)
    print("Note: ↑ means higher is better, ↓ means lower is better")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Image Inpainting Quality Evaluation (pyiqa)")
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Ground truth image directory')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Inpainted image directory')
    parser.add_argument('--save_json', type=str, default=None,
                        help='JSON file path to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Compute device')
    parser.add_argument('--lpips_net', type=str, default='vgg',
                        choices=['vgg', 'alex', 'squeeze'],
                        help='LPIPS network backbone (default: vgg)')
    parser.add_argument('--compute_fid', action='store_true',
                        help='Whether to compute FID (evaluates dataset distribution, not pairwise)')
    parser.add_argument('--local_weights', type=str, default=None,
                        help='Local pretrained weights directory (e.g.: evaluation/checkpoints)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Limit number of images to process (e.g.: 250, only evaluate first 250)')

    args = parser.parse_args()

    # Check directories
    if not os.path.exists(args.gt_dir):
        raise ValueError(f"GT directory does not exist: {args.gt_dir}")
    if not os.path.exists(args.pred_dir):
        raise ValueError(f"Prediction directory does not exist: {args.pred_dir}")

    # Auto-generate JSON filename (concise format, timestamp only)
    if args.save_json is None:
        # Use timestamp to avoid overwriting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save to evaluation directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.save_json = os.path.join(script_dir, f'results_{timestamp}.json')
        print(f"Results will be saved to: {args.save_json}\n")

    # Setup local weights path (default: checkpoints in current directory)
    if args.local_weights is None:
        # Default: checkpoints in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_weights_path = os.path.join(script_dir, 'checkpoints')
        if os.path.exists(default_weights_path):
            args.local_weights = default_weights_path
            print(f"Detected local weights directory: {default_weights_path}\n")

    # Create evaluator
    evaluator = PyIQAEvaluator(
        device=args.device,
        lpips_net=args.lpips_net,
        compute_fid=args.compute_fid,
        local_weights_path=args.local_weights
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        save_json=args.save_json,
        max_images=args.max_images
    )

    # Print results
    print_results(results)


if __name__ == '__main__':
    main()
