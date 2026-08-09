"""
WaveGuideDiff
"""

from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint as nn_checkpoint,
)

import copy
import logging
from os.path import join as pjoin
import torch
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
import scipy.io as sio
import matplotlib.pyplot as plt

# Import base modules
from .model import (
    Mlp,
    WindowAttention,
    window_partition,
    window_reverse,
    PatchEmbed,
    FinalPatchExpand_X4,
)



class DWTForward(nn.Module):
    """
    Haar Wavelet Forward Transform

    Decomposes input into:
    - LL: Low-frequency approximation (structural information)
    - LH: Horizontal high-frequency (horizontal edges)
    - HL: Vertical high-frequency (vertical edges)
    - HH: Diagonal high-frequency (texture details)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            LL: (B, C, H/2, W/2) - Low frequency
            LH: (B, C, H/2, W/2) - Horizontal high frequency
            HL: (B, C, H/2, W/2) - Vertical high frequency
            HH: (B, C, H/2, W/2) - Diagonal high frequency
        """
        # Haar wavelet transform
        x_LL = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] +
                x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2
        x_LH = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] +
                x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 2
        x_HL = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] -
                x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 2
        x_HH = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] -
                x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2

        return x_LL, x_LH, x_HL, x_HH


class DWTInverse(nn.Module):
    """
    Haar Wavelet Inverse Transform

    Reconstructs LL, LH, HL, HH to original resolution
    """
    def __init__(self):
        super().__init__()

    def forward(self, LL, LH, HL, HH):
        """
        Args:
            LL, LH, HL, HH: Each is (B, C, H, W)
        Returns:
            x: (B, C, 2H, 2W) - Reconstructed image
        """
        B, C, H, W = LL.shape

        # Allocate output space
        x = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)

        # Haar inverse transform formula
        x[:, :, 0::2, 0::2] = (LL + LH + HL + HH) / 2
        x[:, :, 0::2, 1::2] = (LL - LH + HL - HH) / 2
        x[:, :, 1::2, 0::2] = (LL + LH - HL - HH) / 2
        x[:, :, 1::2, 1::2] = (LL - LH - HL + HH) / 2

        return x



class FSAM(nn.Module):
    """
    Frequency Selective Attention Module (FSAM) - Module A

    Selectively enhances high-frequency components (LH, HL, HH) decomposed by DWT
    Learns importance of each high-frequency subband through channel attention
    """
    def __init__(self, channels, reduction=4):
        super().__init__()

        # Channel attention - learns importance weights for each high-frequency channel
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial refinement - lightweight depthwise convolution to enhance local features
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, high_freq):
        """
        Args:
            high_freq: (B, 3*C, H, W) - Concatenated LH, HL, HH
        Returns:
            enhanced: (B, 3*C, H, W) - Enhanced high-frequency features
        """
        # Channel attention weighting
        ca_weights = self.channel_attention(high_freq)
        enhanced = high_freq * ca_weights

        # Spatial refinement
        enhanced = enhanced + self.spatial_refine(enhanced)

        return enhanced



class BFIM(nn.Module):
    """
    Bi-directional Frequency Interaction

    Implements bi-directional information exchange between LL (low-frequency) and HF (high-frequency):
    - LL provides structural context to HF
    - HF provides detail information to LL

    Optimized design:
    - First compress HF(3C) to C, then perform interaction, reducing parameters by half
    - Avoids parameter explosion in deep networks (when C=768)

    Key innovation: Two paths share increments, forming complementary enhancement
    """
    def __init__(self, ll_channels, hf_channels, reduction=4):
        """
        Args:
            ll_channels: Number of channels in LL branch (C)
            hf_channels: Number of channels in HF branch (3C, because LH+HL+HH)
        """
        super().__init__()

        self.ll_channels = ll_channels
        self.hf_channels = hf_channels

        # First compress HF to same channel count as LL, reducing subsequent parameters
        # 3C -> C (parameters reduced from 4C to 2C)
        self.hf_compress = nn.Sequential(
            nn.Conv2d(hf_channels, ll_channels, 1, bias=False),
            nn.BatchNorm2d(ll_channels),
            nn.GELU()
        )

        # LL path processing
        self.ll_conv = nn.Sequential(
            nn.Conv2d(ll_channels, ll_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ll_channels),
            nn.GELU()
        )

        # HF path processing (now compressed to C channels)
        self.hf_conv = nn.Sequential(
            nn.Conv2d(ll_channels, ll_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ll_channels),
            nn.GELU()
        )

        # Cross fusion - generate shared increment (now C+C=2C, not C+3C=4C)
        total_channels = ll_channels * 2  # 2C instead of 4C
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // reduction, total_channels, 1, bias=False),
        )

        # Output projection - project fusion result back to LL and HF respectively
        self.ll_proj = nn.Conv2d(total_channels, ll_channels, 1, bias=False)
        # HF needs to expand back to 3C
        self.hf_proj = nn.Conv2d(total_channels, hf_channels, 1, bias=False)

    def forward(self, ll, hf):
        """
        Args:
            ll: (B, C, H, W) - Low-frequency component
            hf: (B, 3C, H, W) - High-frequency component (LH, HL, HH concatenated)
        Returns:
            ll_out: (B, C, H, W) - Enhanced low-frequency
            hf_out: (B, 3C, H, W) - Enhanced high-frequency
        """
        # First compress HF
        hf_compressed = self.hf_compress(hf)  # (B, 3C, H, W) -> (B, C, H, W)

        # Process each path
        ll_feat = self.ll_conv(ll)
        hf_feat = self.hf_conv(hf_compressed)

        # Concatenate and fuse (now 2C instead of 4C)
        cat_feat = torch.cat([ll_feat, hf_feat], dim=1)
        fused = self.fusion_conv(cat_feat)

        # Generate shared increments
        ll_delta = self.ll_proj(fused)
        hf_delta = self.hf_proj(fused)  # Expand back to 3C

        # Residual connection
        ll_out = ll + ll_delta
        hf_out = hf + hf_delta

        return ll_out, hf_out


class FSDC(nn.Module):
    """
    Frequency-aware Spatial Dimension Compression (FSDC)
    LL compressed separately + LH/HL/HH compressed in groups
    Maintains independence of each frequency component, reduces information mixing
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: Input channel count (C)
            out_channels: Output channel count (2C)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # LL compression: C -> C (maintain channel count, preserve structural information)
        self.ll_compress = nn.Linear(in_channels, in_channels, bias=False)

        # HF grouped compression: each high-frequency component C -> C//3
        # Total: 3 * (C//3) = C
        hf_out_per_band = in_channels // 3
        self.lh_compress = nn.Linear(in_channels, hf_out_per_band, bias=False)
        self.hl_compress = nn.Linear(in_channels, hf_out_per_band, bias=False)
        self.hh_compress = nn.Linear(in_channels, hf_out_per_band + in_channels % 3, bias=False)  # Handle remainder

        # Normalization
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, ll, lh, hl, hh):
        """
        Args:
            ll: (B, L, C) - Low frequency
            lh, hl, hh: (B, L, C) - Each high-frequency component
        Returns:
            out: (B, L, 2C) - Compressed features
        """
        # Grouped compression
        ll_out = self.ll_compress(ll)  # (B, L, C)
        lh_out = self.lh_compress(lh)  # (B, L, C//3)
        hl_out = self.hl_compress(hl)  # (B, L, C//3)
        hh_out = self.hh_compress(hh)  # (B, L, C//3 + remainder)

        # Concatenate: LL + LH + HL + HH -> 2C
        out = torch.cat([ll_out, lh_out, hl_out, hh_out], dim=-1)
        out = self.norm(out)

        return out


class FSDCup(nn.Module):
    """
    Grouped Expansion Module - Inverse operation of FSDC

    Used in upsampling path to expand compressed features back to each frequency component
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: Input channel count (2C)
            out_channels: Output channel count (C, channel count per component)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First expand to 4C, then distribute to each component
        self.expand = nn.Linear(in_channels, out_channels * 4, bias=False)
        self.norm = nn.LayerNorm(out_channels * 4)

    def forward(self, x):
        """
        Args:
            x: (B, L, 2C)
        Returns:
            ll, lh, hl, hh: Each is (B, L, C)
        """
        x = self.expand(x)  # (B, L, 4C)
        x = self.norm(x)

        # Split into 4 components
        C = self.out_channels
        ll = x[..., :C]
        lh = x[..., C:2*C]
        hl = x[..., 2*C:3*C]
        hh = x[..., 3*C:]

        return ll, lh, hl, hh


class MSR(nn.Module):
    """
    Multi-Scale Residual Module (MSR)

    Provides a shortcut bypassing complex processing during downsampling
    Ensures original information is not completely lost during compression
    """
    def __init__(self, in_channels, out_channels, input_resolution):
        """
        Args:
            in_channels: Input channel count (C)
            out_channels: Output channel count (2C)
            input_resolution: Input resolution (H, W)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_resolution = input_resolution

        # Spatial downsampling + channel expansion
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """
        Args:
            x: (B, H*W, C) - Sequence format input
        Returns:
            out: (B, H/2*W/2, 2C) - Residual features
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        # Convert to image format
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # Downsample
        x = self.downsample(x)  # (B, 2C, H/2, W/2)

        # Convert back to sequence format
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, self.out_channels)

        return x


class MSRup(nn.Module):
    """
    Global Residual Upsampling Module - Symmetric version of MSR

    Used in upsampling path to protect information during reconstruction
    """
    def __init__(self, in_channels, out_channels, input_resolution):
        """
        Args:
            in_channels: Input channel count (2C)
            out_channels: Output channel count (C)
            input_resolution: Input resolution (H, W)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_resolution = input_resolution

        # Channel compression + spatial upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        """
        Args:
            x: (B, H*W, 2C) - Sequence format input
        Returns:
            out: (B, 4*H*W, C) - Residual features
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        # Convert to image format
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, 2C, H, W)

        # Upsample
        x = self.upsample(x)  # (B, C, 2H, 2W)

        # Convert back to sequence format
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, self.out_channels)

        return x




class DWTEncoding(nn.Module):
    """
    Wavelet Transform Downsampling Module

    Input: (B, H*W, C)
    Output: (B, H/2*W/2, 2C)
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        # 1. DWT transform
        self.dwt = DWTForward()

        # 2. Frequency Selective Attention Module (FSAM)
        self.fsam = FSAM(dim * 3, reduction=4)

        # 3. Bi-directional interaction module
        self.bfim = BFIM(
            ll_channels=dim,
            hf_channels=dim * 3,
            reduction=4
        )

        # 4. Grouped compression
        self.fsdc = FSDC(
            in_channels=dim,
            out_channels=dim * 2
        )

        # 5. Global residual
        self.msr = MSR(
            in_channels=dim,
            out_channels=dim * 2,
            input_resolution=input_resolution
        )

        # Fusion weight (learnable)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x: (B, H*W, C) - Sequence format input
        Returns:
            out: (B, H/2*W/2, 2C) - Downsampled sequence
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input size {L} doesn't match resolution {H}x{W}"
        assert H % 2 == 0 and W % 2 == 0, f"Resolution ({H}x{W}) must be even"

        # Save original input for global residual
        x_residual = self.msr(x)  # (B, H/2*W/2, 2C)

        # 1. Convert to image format (B, C, H, W)
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 2. DWT decomposition
        LL, LH, HL, HH = self.dwt(x_img)  # Each is (B, C, H/2, W/2)

        # 3. High-frequency enhancement (HFE)
        high_freq = torch.cat([LH, HL, HH], dim=1)  # (B, 3C, H/2, W/2)
        high_freq_enhanced = self.fsam(high_freq)

        # 4. Bi-directional interaction
        LL_inter, HF_inter = self.bfim(LL, high_freq_enhanced)

        # Separate enhanced high-frequency components
        LH_e, HL_e, HH_e = torch.chunk(HF_inter, 3, dim=1)

        # 5. Convert back to sequence format for grouped compression
        H_new, W_new = H // 2, W // 2
        LL_seq = LL_inter.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        LH_seq = LH_e.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        HL_seq = HL_e.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        HH_seq = HH_e.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

        # 6. Grouped compression
        x_compressed = self.fsdc(LL_seq, LH_seq, HL_seq, HH_seq)  # (B, L/4, 2C)

        # 7. Global residual fusion
        weight = torch.sigmoid(self.fusion_weight)
        out = weight * x_compressed + (1 - weight) * x_residual

        return out

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class IDWTExpanding(nn.Module):
    """
    Inverse Wavelet Transform Upsampling Module


    Input: (B, H*W, C)
    Output: (B, 4*H*W, C/2)
    """
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale

        # Output channel count
        out_dim = dim // dim_scale

        # 1. Grouped expansion
        self.grouped_expand = FSDCup(
            in_channels=dim,
            out_channels=out_dim
        )

        # 2. Bi-directional frequency interaction (BFIM) - let network learn information exchange between LL and HF
        self.bfim = BFIM(
            ll_channels=out_dim,
            hf_channels=out_dim * 3,
            reduction=4
        )

        # 3. IDWT transform
        self.idwt = DWTInverse()

        # 4. Multi-scale residual (MSR)
        self.msr = MSRup(
            in_channels=dim,
            out_channels=out_dim,
            input_resolution=input_resolution
        )

        # Fusion weight (learnable)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # Normalization
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        """
        Args:
            x: (B, H*W, C)
        Returns:
            out: (B, 4*H*W, C/2)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input size {L} doesn't match resolution {H}x{W}"

        out_dim = C // self.dim_scale

        # Save original input for multi-scale residual
        x_residual = self.msr(x)  # (B, 4*H*W, C/2)

        # 1. Grouped expansion
        LL_seq, LH_seq, HL_seq, HH_seq = self.grouped_expand(x)  # Each is (B, L, C/2)

        # 2. Convert to image format
        LL = LL_seq.view(B, H, W, out_dim).permute(0, 3, 1, 2).contiguous()
        LH = LH_seq.view(B, H, W, out_dim).permute(0, 3, 1, 2).contiguous()
        HL = HL_seq.view(B, H, W, out_dim).permute(0, 3, 1, 2).contiguous()
        HH = HH_seq.view(B, H, W, out_dim).permute(0, 3, 1, 2).contiguous()

        # 3. Bi-directional frequency interaction (BFIM) - let network learn how to distribute information to each subband
        high_freq = torch.cat([LH, HL, HH], dim=1)
        LL_inter, HF_inter = self.bfim(LL, high_freq)

        # 4. Directly separate high-frequency components (no longer apply HFR enhancement)
        LH_out, HL_out, HH_out = torch.chunk(HF_inter, 3, dim=1)

        # 5. IDWT reconstruction
        x_recon = self.idwt(LL_inter, LH_out, HL_out, HH_out)  # (B, C/2, 2H, 2W)

        # 6. Convert back to sequence format
        x_main = x_recon.permute(0, 2, 3, 1).contiguous().view(B, -1, out_dim)

        # 7. Multi-scale residual fusion (MSR)
        weight = torch.sigmoid(self.fusion_weight)
        out = weight * x_main + (1 - weight) * x_residual

        # 8. Normalization
        out = self.norm(out)

        return out

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"




class SAFM(nn.Module):
    """
    Spatial-Aware Feature Merging Module (SAFM) - Module C

    Adaptively fuses encoder and decoder features through channel attention
    Used for intelligent fusion of skip connections
    """
    def __init__(self, dim, height=2, reduction=8):
        super().__init__()
        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, decoder_feat, encoder_feat):
        """
        Args:
            decoder_feat: (B, L, C) - Decoder features
            encoder_feat: (B, L, C) - Encoder skip connection features
        Returns:
            fused: (B, L, C) - Fused features
        """
        B, L, C = decoder_feat.shape
        H = W = int(L ** 0.5)

        # Convert to image format
        f1 = decoder_feat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        f2 = encoder_feat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Stack: (B, 2, C, H, W)
        in_feats = torch.stack([f1, f2], dim=1)

        # Compute attention weights
        feats_sum = torch.sum(in_feats, dim=1)  # (B, C, H, W)
        attn = self.mlp(self.avg_pool(feats_sum))  # (B, 2C, 1, 1)
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))  # (B, 2, C, 1, 1)

        # Weighted fusion
        out = torch.sum(in_feats * attn, dim=1)  # (B, C, H, W)

        # Convert back to sequence format
        out = out.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        return out




class TimestepBlock(nn.Module):
    """Base class for modules that support time embedding"""
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential that supports time embedding"""
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class SwinTransformerBlock(TimestepBlock):
    """Swin Transformer Block"""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, time_embed_dim=128 * 4, out_channels=64):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = False

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.out_channels = out_channels
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(time_embed_dim, self.dim),
        )

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            _, mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, emb):
        return nn_checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x.clone()
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        nW, x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(B, -1, self.window_size * self.window_size, C)

        # Add time embedding
        time_embed = self.emb_layers(emb).unsqueeze(1).unsqueeze(1)
        time_embed = time_embed.repeat(1, nW, 1, 1)
        x_windows = torch.concat((x_windows, time_embed), dim=-2)
        x_windows = x_windows.flatten(0, 1)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        h = x.view(B, H * W, C)

        # FFN
        drop_path = self.drop_path(h)
        x = shortcut + drop_path
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(TimestepBlock):
    """Basic Swin Transformer layer (downsampling path) - uses DWT downsampling"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 time_embed_dim=128 * 4, out_channels=64):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = False

        # Build Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, time_embed_dim=time_embed_dim, out_channels=out_channels)
            for i in range(depth)
        ])

        # Use DWT downsampling
        if downsample is not None:
            self.downsample = TimestepEmbedSequential(
                DWTEncoding(input_resolution, dim=dim, norm_layer=norm_layer)
            )
        else:
            self.downsample = None

    def forward(self, x, emb):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, emb)
            else:
                x = blk(x, emb)
        if self.downsample is not None:
            x = self.downsample(x, emb)
        return x


class BasicLayer_up(TimestepBlock):
    """Basic Swin Transformer layer (upsampling path) - uses IDWT upsampling"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 out_channels=64, time_embed_dim=128 * 4):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = False

        # Build Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, out_channels=out_channels, time_embed_dim=time_embed_dim)
            for i in range(depth)
        ])

        # Use IDWT upsampling
        if upsample is not None:
            self.upsample = TimestepEmbedSequential(
                IDWTExpanding(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            )
        else:
            self.upsample = None

    def forward(self, x, emb):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, emb)
            else:
                x = blk(x, emb)
        if self.upsample is not None:
            x = self.upsample(x, emb)
        return x



class WaveGuideDiff(nn.Module):

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            img_size=512,
            patch_size=4,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            depths_decoder=[1, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            final_upsample="expand_first",
            use_safm=True,  # Added for compatibility with script_util.py
            use_msr=True,   # Added for compatibility with script_util.py
            **kwargs,  # Accept any additional parameters without error
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.channel_mult = [2, 2, 2, 2]
        self.num_layers = len(channel_mult)
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = False
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.depths_decoder = depths_decoder
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.ape = ape
        self.patch_norm = patch_norm
        self.final_upsample = final_upsample

        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=self.in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Build downsampling path (Encoder)
        self.input_blocks = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        num_res_blocks = 1

        for i_layer, mult in enumerate(self.depths):
            for _ in range(num_res_blocks):
                layers = BasicLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                      patches_resolution[1] // (2 ** i_layer)),
                    depth=self.depths[i_layer],
                    num_heads=self.num_heads[i_layer],
                    window_size=self.window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                    drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                    drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=DWTEncoding if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    out_channels=max(1, int(64 / (pow(2, i_layer)))),
                    time_embed_dim=time_embed_dim
                )
                self.input_blocks.append(TimestepEmbedSequential(layers))

        # Build upsampling path (Decoder)
        self.output_blocks_layers_up = nn.ModuleList([])
        self.safm_modules = nn.ModuleList([])

        for i_layer, mult in enumerate(self.depths):
            for _ in range(num_res_blocks):
                # Skip connection fusion layer - using SAFM
                if i_layer > 0:
                    current_dim = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
                    self.safm_modules.append(SAFM(current_dim, height=2, reduction=8))
                else:
                    self.safm_modules.append(None)

                # Upsampling layer
                if i_layer == 0:
                    layer_up = IDWTExpanding(
                        input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                        dim_scale=2, norm_layer=norm_layer)
                else:
                    layer_up = BasicLayer_up(
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                        input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        depth=depths[(self.num_layers - 1 - i_layer)],
                        num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                        window_size=self.window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                            depths[:(self.num_layers - 1 - i_layer) + 1])],
                        norm_layer=norm_layer,
                        upsample=IDWTExpanding if (i_layer < self.num_layers - 1) else None,
                        use_checkpoint=use_checkpoint,
                        out_channels=max(1, int(64 / (pow(2, 3 - i_layer)))),
                        time_embed_dim=time_embed_dim
                    )
                self.output_blocks_layers_up.append(TimestepEmbedSequential(layer_up))

        # Normalization layers
        self.norm = nn.LayerNorm(self.num_features)
        self.norm_up = nn.LayerNorm(self.embed_dim)

        # Final upsampling and output
        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim_scale=4, dim=embed_dim)
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=embed_dim, out_channels=self.out_channels, kernel_size=1, bias=False))

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks_layers_up.apply(convert_module_to_f16)
        self.safm_modules.apply(convert_module_to_f16)
        self.norm.apply(convert_module_to_f16)
        self.norm_up.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks_layers_up.apply(convert_module_to_f32)
        self.safm_modules.apply(convert_module_to_f32)
        self.norm.apply(convert_module_to_f32)
        self.norm_up.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None, **kwargs):
        """
        Forward propagation

        Args:
            x: (B, C, H, W) Input image
            timesteps: (B,) Timesteps
            y: (B,) Class labels (optional)
        Returns:
            out: (B, C, H, W) Output image
        """
        # Patch Embedding
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"

        # Time embedding
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)

        # Encoder (downsampling path)
        for module in self.input_blocks:
            hs.append(h)
            h = module(h, emb)

        h = self.norm(h)

        # Decoder (upsampling path)
        for inx, layer_up in enumerate(self.output_blocks_layers_up):
            if inx == 0:
                h = layer_up(h, emb)
            else:
                skip = hs[3 - inx]

                # Use SAFM for skip connection fusion
                h = self.safm_modules[inx](h, skip)

                h = layer_up(h, emb)

        h = h.type(x.dtype)
        h = self.norm_up(h)

        # Final upsampling
        H, W = self.patches_resolution
        B, L, C = h.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            h = self.up(h)
            h = h.view(B, 4 * H, 4 * W, -1)
            h = h.permute(0, 3, 1, 2)  # B,C,H,W
            h = self.output(h)

        return h

    def get_feature_vectors(self, x, timesteps, y=None):
        """Get intermediate features (for visualization or analysis)"""
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)

        for inx, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
            if inx != len(self.input_blocks) - 1:
                result["down"].append(h.type(x.dtype))
            else:
                result["middle"] = h.type(x.dtype)

        for inx, layer_up in enumerate(self.output_blocks_layers_up):
            if inx == 0:
                h = layer_up(h, emb)
            else:
                skip = hs[3 - inx]
                h = self.safm_modules[inx](h, skip)
                h = layer_up(h, emb)
            result["up"].append(h.type(x.dtype))

        return result


