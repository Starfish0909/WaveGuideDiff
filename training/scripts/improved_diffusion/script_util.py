import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .waveguidediff import WaveGuideDiff

NUM_CLASSES = 0


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=512,
        num_channels=128,
        num_res_blocks=2,
        num_heads=[3, 6, 12, 24],  # Default for WaveGuideDiff (UNet/DiT will use first element)
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
        use_checkpoint=False,
        use_scale_shift_norm=True,
        num_in_channels=3,
        num_out_channels=6,
        model_type="waveguidediff",  # Default to WaveGuideDiff
        # WaveGuideDiff specific parameters
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_safm=True,
        use_msr=True,
        # DiT specific parameters
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
        model_type="waveguidediff",
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
        model_type="waveguidediff",
        **kwargs,
):
    """
    Create a model based on model_type.
    """
    model_type_lower = model_type.lower()


    if model_type_lower == "waveguidediff":
        # Extract WaveGuideDiff-specific parameters
        patch_size = kwargs.get('patch_size', 4)
        embed_dim = kwargs.get('embed_dim', 96)
        depths = kwargs.get('depths', [2, 2, 2, 2])
        window_size = kwargs.get('window_size', 8)
        mlp_ratio = kwargs.get('mlp_ratio', 4.0)
        qkv_bias = kwargs.get('qkv_bias', True)
        drop_rate = kwargs.get('drop_rate', 0.0)
        drop_path_rate = kwargs.get('drop_path_rate', 0.1)
        ape = kwargs.get('ape', False)
        patch_norm = kwargs.get('patch_norm', True)
        use_safm = kwargs.get('use_safm', True)
        use_msr = kwargs.get('use_msr', True)

        # Handle num_heads parameter
        if isinstance(num_heads, int):
            num_heads_list = [num_heads] * len(depths)
        elif isinstance(num_heads, list):
            num_heads_list = num_heads
        else:
            num_heads_list = [3, 6, 12, 24]

        return WaveGuideDiff(
            img_size=image_size,
            patch_size=patch_size,
            in_channels=num_in_channels,
            model_channels=num_channels,
            out_channels=num_out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads_list,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            use_safm=use_safm,
            use_msr=use_msr,
        )


    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: waveguidediff"
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
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
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
            # Handle list parameters (depths, num_heads)
            if k in ['depths', 'num_heads']:
                v_type = lambda x: [int(i) for i in x.split(',')] if isinstance(x, str) else x
            else:
                v_type = lambda x: x if isinstance(x, list) else [int(i) for i in x.split(',')]
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
