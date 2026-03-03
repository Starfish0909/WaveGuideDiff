"""
Train a diffusion model on images.
"""

import argparse
import random
import numpy as np
import torch

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def set_seed(seed, strict_mode=True):
    """
    Fix random seed for reproducibility

    Args:
        seed (int): Random seed value
        strict_mode (bool): Whether to use strict mode
            - True: Fully reproducible, but training speed reduced by 10-20%
            - False: Only fixes initialization, no speed impact, but may have minor differences
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict_mode:
        # Strict mode: fully reproducible, but 10-20% slower
        # Reason:
        #   1. cuDNN benchmark=False disables auto algorithm optimization
        #   2. deterministic=True forces deterministic algorithms, sacrificing parallelism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Relaxed mode: only fixes initialization, allows cuDNN optimization
        # - Same weight initialization
        # - Same data loading order
        # - But cuDNN algorithms may differ slightly (diff < 1e-5)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def main():
    args = create_argparser().parse_args()

    # Fix random seed (if seed parameter is specified)
    if hasattr(args, 'seed') and args.seed is not None:
        set_seed(args.seed)
        logger.log(f"Random seed set to: {args.seed}")
        logger.log(f"   - Ensures reproducibility for fair model comparison")
        logger.log(f"   - Note: May reduce training speed by ~10-20%")
    else:
        logger.log("No seed specified - training with random initialization")
        logger.log("   - Consider using --seed for reproducible experiments")

    dist_util.setup_dist()

    # If resume checkpoint specified, set env var for logger to detect resume mode
    if args.resume_checkpoint:
        import os
        os.environ["RESUME_CHECKPOINT"] = args.resume_checkpoint
    
    logger.configure()

    logger.log("=" * 60)
    logger.log("Training configuration:")
    logger.log(f"  model_type: {args.model_type}")
    logger.log(f"  noise_schedule: {args.noise_schedule}")
    logger.log(f"  diffusion_steps: {args.diffusion_steps}")
    logger.log(f"  image_size: {args.image_size}")
    logger.log(f"  lr: {args.lr}")
    logger.log(f"  batch_size: {args.batch_size}")
    logger.log("=" * 60)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")

    # Build model config dict (passed to visualize_with_mask.py for visualization)
    # Visualization diffusion_steps can be set independently (via --viz_diffusion_steps)
    viz_diffusion_steps = getattr(args, 'viz_diffusion_steps', None)
    if viz_diffusion_steps is None or viz_diffusion_steps == '':
        # Default to 1000 steps if not specified
        viz_diffusion_steps = 1000
    else:
        # Convert to integer
        try:
            viz_diffusion_steps = int(viz_diffusion_steps)
        except:
            logger.log(f"Warning: cannot parse viz_diffusion_steps='{viz_diffusion_steps}', using default 1000")
            viz_diffusion_steps = 1000

    model_config = {
        # Model architecture parameters
        'image_size': args.image_size,
        'num_channels': args.num_channels,
        'num_res_blocks': args.num_res_blocks,
        'learn_sigma': args.learn_sigma,
        'num_in_channels': getattr(args, 'num_in_channels', 3),
        'num_out_channels': getattr(args, 'num_out_channels', 6),
        'model_type': getattr(args, 'model_type', 'unet'),
        # DiT-specific parameters
        'dit_depth': getattr(args, 'dit_depth', 12),
        'dit_hidden_size': getattr(args, 'dit_hidden_size', 768),
        'dit_patch_size': getattr(args, 'dit_patch_size', 8),
        'dit_mlp_ratio': getattr(args, 'dit_mlp_ratio', 4.0),
        # Diffusion parameters (critical)
        # Note: visualization can use different steps than training (via --viz_diffusion_steps)
        # Training can use more steps (e.g. 4000), but visualization uses fewer for speed
        'diffusion_steps': viz_diffusion_steps,
        'noise_schedule': args.noise_schedule,
    }

    logger.log("Model config (for visualization):")
    for key, value in model_config.items():
        logger.log(f"  {key}: {value}")
    logger.log(f"  Training uses {args.diffusion_steps} steps, visualization uses {viz_diffusion_steps} steps")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        lr_warmup_steps=args.lr_warmup_steps,  # Learning rate warmup steps
        viz_gt_dir=getattr(args, 'viz_gt_dir', None),
        viz_mask_dir=getattr(args, 'viz_mask_dir', None),
        viz_gt_path=getattr(args, 'viz_gt_path', None),
        viz_mask_path=getattr(args, 'viz_mask_path', None),
        viz_respacing=getattr(args, 'viz_respacing', None),
        viz_schedule_jump_params=getattr(args, 'viz_schedule_jump_params', None),
        viz_output_dir=getattr(args, 'viz_output_dir', None),
        model_config=model_config,  # Pass model config 
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        lr_scheduler="linear",  # Learning rate scheduler: "linear", "cosine", or "cosine_warmup"
        lr_min=0.0,  # Minimum learning rate (for cosine annealing)
        lr_warmup_steps=0,  # Learning rate warmup steps (e.g. 5000)
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        viz_gt_dir="",  # GT image directory (for visualization)
        viz_mask_dir="",  # Mask directory (for visualization)
        viz_gt_path="",  # Fixed GT image path (for visualization)
        viz_mask_path="",  # Fixed mask path (for visualization)
        viz_respacing="",  # Timestep respacing for visualization
        viz_schedule_jump_params="",  # Schedule jump params (dict as string)
        viz_diffusion_steps="",  # Diffusion steps for visualization (independent of training, default 1000)
        viz_output_dir="",  # Visualization output directory name
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    # Add seed parameter separately with explicit int type
    parser.add_argument("--seed", type=int, default=None, help="Random seed (for reproducibility)")
    return parser


if __name__ == "__main__":
    main()
