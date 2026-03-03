"""
Shared model registry - add new models here, both training and inference will auto-detect them

Usage:
1. Add new model config in MODEL_CONFIGS
2. Model files go in training/scripts/improved_diffusion/
3. Done! Both training and inference will auto-detect

Config format:
    'model_key': {
        'filename': 'model_file.py',
        'class_name': 'ModelClassName',
        'aliases': ['alias1', 'alias2'],  # optional
        'params': ['param1', 'param2'],  # optional: specific params to extract from kwargs
        'param_defaults': {'param1': default_value},  # optional: parameter defaults
    }
"""

import importlib
import importlib.util
import os
import sys

MODEL_CONFIGS = {
    # SwinUNet - Base Swin Transformer UNet
    'swin_unet': {
        'filename': 'swin_unet.py',
        'class_name': 'SwinUNetModel',
        'aliases': ['swinunet', 'swin'],
        'param_mapping': {'img_size': 'sample_size'},  # img_size -> sample_size
    },

    # WaveGuideDiff - Wavelet-guided diffusion model
    'waveguidediff': {
        'filename': 'waveguidediff.py',
        'class_name': 'WaveGuideDiff',
        'aliases': ['wave_guide_diff', 'wgd', 'waveguide'],
        'params': ['patch_size', 'embed_dim', 'depths', 'window_size', 'mlp_ratio',
                   'qkv_bias', 'drop_rate', 'drop_path_rate', 'ape', 'patch_norm',
                   'use_safm', 'use_msr'],
        'param_defaults': {
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 2, 2],
            'window_size': 8,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'ape': False,
            'patch_norm': True,
            'use_safm': True,
            'use_msr': True,
        },
    },


}


class ModelRegistry:
    """Model registry - dynamically load and create models"""

    MODEL_DIR = 'training/scripts/improved_diffusion'

    def __init__(self, project_root=None):
        """
        Args:
            project_root: Project root path. If None, auto-detect.
        """
        if project_root is None:
            # Auto-detect project root (parent of shared directory)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = project_root
        self.model_dir = os.path.join(project_root, self.MODEL_DIR)
        self._loaded_classes = {}
        self._alias_map = self._build_alias_map()

    def _build_alias_map(self):
        """Build alias-to-key mapping"""
        alias_map = {}
        for key, config in MODEL_CONFIGS.items():
            alias_map[key.lower()] = key
            for alias in config.get('aliases', []):
                alias_map[alias.lower()] = key
        return alias_map

    def _resolve_model_key(self, model_type):
        """Resolve model type, supports aliases"""
        return self._alias_map.get(model_type.lower())

    def _load_model_class(self, model_key):
        """Dynamically load model class (always from training directory)"""
        if model_key in self._loaded_classes:
            return self._loaded_classes[model_key]

        config = MODEL_CONFIGS.get(model_key)
        if not config:
            return None

        try:
            module_name = config['filename'][:-3]  # Remove .py extension
            file_path = os.path.join(self.model_dir, config['filename'])

            if not os.path.exists(file_path):
                print(f"Model file not found: {file_path}")
                return None

            # Ensure training/scripts is in sys.path
            scripts_dir = os.path.dirname(self.model_dir)  # training/scripts
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)

            # Use package import (supports relative imports)
            full_module_name = f"improved_diffusion.{module_name}"
            module = importlib.import_module(full_module_name)

            model_class = getattr(module, config['class_name'])
            self._loaded_classes[model_key] = model_class
            return model_class
        except Exception as e:
            print(f"Failed to load model {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_supported_models(self):
        """Get all supported model types"""
        return list(MODEL_CONFIGS.keys())

    def get_model_config(self, model_type):
        """Get model config"""
        model_key = self._resolve_model_key(model_type)
        if model_key:
            return MODEL_CONFIGS.get(model_key)
        return None

    def create_model(
        self,
        model_type,
        # Common parameters
        in_channels,
        model_channels,
        out_channels,
        img_size,
        dropout,
        channel_mult,
        num_classes,
        use_checkpoint,
        num_heads,
        num_heads_upsample,
        use_scale_shift_norm,
        # UNet-specific parameters
        num_res_blocks=None,
        attention_resolutions=None,
        # Other parameters
        **kwargs
    ):
        """
        Create model instance by model_type

        Returns:
            model: Model instance

        Raises:
            ValueError: If model_type is not supported
        """
        model_key = self._resolve_model_key(model_type)
        if not model_key:
            supported = ', '.join(self.get_supported_models())
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {supported}")

        config = MODEL_CONFIGS[model_key]

        # Special handling for UNet
        if config.get('special_handler') == 'unet':
            return self._create_unet(
                in_channels, model_channels, out_channels,
                num_res_blocks, attention_resolutions, dropout,
                channel_mult, num_classes, use_checkpoint,
                num_heads, num_heads_upsample, use_scale_shift_norm
            )

        # Load model class
        model_class = self._load_model_class(model_key)
        if not model_class:
            raise ValueError(f"Failed to load model class: {model_key}")

        # Build model arguments
        model_kwargs = {
            'in_channels': in_channels,
            'model_channels': model_channels,
            'out_channels': out_channels,
            'img_size': img_size,
            'dropout': dropout,
            'channel_mult': channel_mult,
            'num_classes': num_classes,
            'use_checkpoint': use_checkpoint,
            'num_heads': num_heads,
            'num_heads_upsample': num_heads_upsample,
            'use_scale_shift_norm': use_scale_shift_norm,
        }

        # Handle parameter mapping (e.g. img_size -> sample_size)
        param_mapping = config.get('param_mapping', {})
        for old_name, new_name in param_mapping.items():
            if old_name in model_kwargs:
                model_kwargs[new_name] = model_kwargs.pop(old_name)

        # Add model-specific parameters
        param_defaults = config.get('param_defaults', {})
        for param in config.get('params', []):
            model_kwargs[param] = kwargs.get(param, param_defaults.get(param))

        return model_class(**model_kwargs)

    def _create_unet(
        self, in_channels, model_channels, out_channels,
        num_res_blocks, attention_resolutions, dropout,
        channel_mult, num_classes, use_checkpoint,
        num_heads, num_heads_upsample, use_scale_shift_norm
    ):
        """Create UNet model (requires special handling)"""
        model_class = self._load_model_class('unet')
        if not model_class:
            raise ValueError("Failed to load UNetModel")

        # UNet uses fixed number of attention heads
        unet_num_heads = 1
        return model_class(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            num_heads=unet_num_heads,
            num_heads_upsample=unet_num_heads,
            use_scale_shift_norm=use_scale_shift_norm,
        )


def get_channel_mult(image_size, model_type):
    """Get channel_mult based on image size and model type

    UNet config reference from OpenAI Improved DDPM / ADM official config:
    - 512x512: channel_mult=(1,1,2,2,4,4), downsample 5 times to 16x16
    - 256x256: channel_mult=(1,1,2,2,4,4), downsample 5 times to 8x8

    Swin Transformer series models use simpler config:
    - Use (1,2,4) since Patch Embedding already does initial downsampling
    """
    # UNet uses official config
    if model_type.lower() == "unet":
        unet_channel_mult_map = {
            512: (1, 1, 2, 2, 4, 4),  # Official 512x512 config
            256: (1, 1, 2, 2, 4, 4),  # Official 256x256 config
            128: (1, 1, 2, 2, 4, 4),
            64: (1, 2, 3, 4),
            32: (1, 2, 2, 2),
        }
        if image_size not in unet_channel_mult_map:
            raise ValueError(f"Unsupported image size for UNet: {image_size}")
        return unet_channel_mult_map[image_size]

    # DiT/UViT use default values
    if model_type.lower() in ["dit", "uvit"]:
        return (1, 1, 2, 2, 4, 4)

    # SegFormer uses same config as Swin series
    # (SegFormer controls channels via dims parameter, channel_mult mainly for compatibility)

    # Swin Transformer series models use simplified config
    swin_channel_mult_map = {
        512: (1, 2, 4),
        256: (1, 2, 4),
        128: (1, 2, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 2, 2),
    }

    if image_size not in swin_channel_mult_map:
        raise ValueError(f"Unsupported image size: {image_size}")

    return swin_channel_mult_map[image_size]


def get_model_defaults():
    """Get model default parameters (shared by training and inference)"""
    return dict(
        image_size=512,
        num_channels=128,
        num_res_blocks=2,
        num_heads=[3, 6, 12, 24],
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.0,
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        num_in_channels=3,
        num_out_channels=6,
        model_type="swin_unet",
        # DiT parameters
        dit_depth=12,
        dit_hidden_size=768,
        dit_patch_size=8,
        dit_mlp_ratio=4.0,
    )
