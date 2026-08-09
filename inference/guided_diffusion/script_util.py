"""
Inference-side script_util.py
Model files are automatically loaded from training/scripts/improved_diffusion/
To add new models, just modify shared/model_registry.py!
"""
import argparse
import inspect
import sys
import os

# Add shared directory to Python path
_shared_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
if _shared_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_shared_path))

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

# Use shared model registry - to add new models, just modify shared/model_registry.py!
from model_registry import ModelRegistry, get_channel_mult, get_model_defaults, MODEL_CONFIGS

# Initialize model registry (automatically loads models from training directory)
_model_registry = ModelRegistry()

NUM_CLASSES = 0


def model_and_diffusion_defaults():
    """Defaults for image training."""
    return dict(
        image_size=512,
        num_channels=128,
        num_res_blocks=2,
        num_heads=[3, 6, 12, 24],
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.0,
        learn_sigma=True,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=4000,
        noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=True,
        use_scale_shift_norm=True,
        num_in_channels=3,
        num_out_channels=6,
        model_type="swin_unet",
        # SwinUNet / WaveGuideDiff parameters
        embed_dim=96,
        depths=[2, 2, 2, 2],
        patch_size=4,
        mlp_ratio=4.0,
        window_size=8,
        # DiT parameters
        dit_depth=12,
        dit_hidden_size=768,
        dit_patch_size=8,
        dit_mlp_ratio=4.0,
    )


def create_model_and_diffusion(
        image_size,
        class_cond,
        learn_sigma,
        sigma_small,
        num_channels,
        num_res_blocks,
        num_heads,
        num_heads_upsample,
        attention_resolutions,
        dropout,
        diffusion_steps,
        noise_schedule,
        timestep_respacing,
        use_kl,
        predict_xstart,
        rescale_timesteps,
        rescale_learned_sigmas,
        use_checkpoint,
        use_scale_shift_norm,
        num_in_channels,
        num_out_channels,
        model_type="swin_unet",
        conf=None,
        **kwargs,
):
    print(f"Creating {model_type} model with:")
    print(f"  - Image size: {image_size}")
    print(f"  - Learn sigma: {learn_sigma}")
    print(f"  - Class conditioning: {class_cond}")
    print(f"  - Output channels: {num_out_channels}")

    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        num_in_channels=num_in_channels,
        num_out_channels=num_out_channels,
        model_type=model_type,
        **kwargs,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        conf=conf,
    )
    return model, diffusion


def create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma,
        class_cond,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
        num_in_channels,
        num_out_channels,
        model_type="swin_unet",
        **kwargs,
):
    """
    Create model using shared model registry.
    Model files are automatically loaded from training/scripts/improved_diffusion/!
    """
    # Get channel_mult
    channel_mult = get_channel_mult(image_size, model_type)

    # Process attention_resolutions (required for UNet)
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # Create model using model registry
    return _model_registry.create_model(
        model_type=model_type,
        in_channels=num_in_channels,
        model_channels=num_channels,
        out_channels=num_out_channels,
        img_size=image_size,
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        **kwargs,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 512
    res["small_size"] = 128
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
        large_size,
        small_size,
        class_cond,
        learn_sigma,
        num_channels,
        num_res_blocks,
        num_heads,
        num_heads_upsample,
        attention_resolutions,
        dropout,
        diffusion_steps,
        noise_schedule,
        timestep_respacing,
        use_kl,
        predict_xstart,
        rescale_timesteps,
        rescale_learned_sigmas,
        use_checkpoint,
        use_scale_shift_norm,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma,
        class_cond,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
):
    _ = small_size

    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4, 8)
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    # Load SuperResModel from training directory (using package import)
    from model_registry import ModelRegistry
    registry = ModelRegistry()
    scripts_dir = os.path.dirname(registry.model_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from improved_diffusion.unet import SuperResModel

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
        conf=None,
        **kwargs,  # Ignore training-specific parameters
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps, use_scale=True)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        conf=conf,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        elif isinstance(v, list):
            if k == 'depths':
                v_type = lambda x: [int(i) for i in x.split(',')] if isinstance(x, str) else x
            elif k == 'num_heads':
                v_type = lambda x: [int(i) for i in x.split(',')] if isinstance(x, str) else x
            else:
                v_type = lambda x: x if isinstance(x, list) else [int(i) for i in x.split(',')]
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
