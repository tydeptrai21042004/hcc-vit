#!/usr/bin/env python3
"""
ViT adapter block with HCC token adapter support.
"""
from functools import partial

import torch
import torch.nn as nn

from timm.models.layers import Mlp, DropPath
from timm.models.vision_transformer import Block  # base ViT block

# project logger (avoid clobbering stdlib `logging`)
from ...utils import logging as vlogging
logger = vlogging.get_logger("visual_prompt")

# src/models/vit_adapter/hcc_adapter.py
from .hcc_adapter import HCCTokenAdapter


def build_adapter(name, embed_dim, grid_size, cfg):
    """
    Factory for adapters used inside ViT blocks (token-space adapters).
    """
    name = (name or "").lower()
    if name == "hcc":
        hcc = cfg.ADAPTER.HCC
        return HCCTokenAdapter(
            embed_dim=embed_dim,
            grid_size=grid_size,                # (H, W) of patch grid
            M=getattr(hcc, "M", 1),
            h=getattr(hcc, "H", 1),
            axis=getattr(hcc, "AXIS", "hw"),
            alpha_group=getattr(hcc, "ALPHA_GROUP", 16),
            per_channel=getattr(hcc, "PER_CHANNEL", None),
            tie_sym=getattr(hcc, "TIE_SYM", True),
            # Prefer the new NO_PW key. For old configs that only define USE_PW,
            # convert USE_PW=True -> NO_PW=False.
            no_pw=getattr(hcc, "NO_PW", not getattr(hcc, "USE_PW", False)),
            pw_ratio=getattr(hcc, "PW_RATIO", 32),
            pw_groups=getattr(hcc, "PW_GROUPS", 4),
            use_bn=getattr(hcc, "USE_BN", False),
            residual_scale=getattr(hcc, "RESIDUAL_SCALE", 1.0),
            gate_init=getattr(hcc, "GATE_INIT", 0.0),
            padding_mode=getattr(hcc, "PADDING", "reflect"),
            dilations=getattr(hcc, "DILATIONS", None),
            scale_adaptive=getattr(hcc, "SCALE_ADAPTIVE", False),
            separate_axis_kernels=getattr(hcc, "SEPARATE_AXIS_KERNELS", True),
            gate_temperature=getattr(hcc, "GATE_TEMPERATURE", 1.0),
            input_adaptive_gate=getattr(hcc, "INPUT_ADAPTIVE_GATE", False),
            gate_reduction=getattr(hcc, "GATE_REDUCTION", 4),
        )
    # Add more adapters here if needed (e.g., Pfeiffer, LoRA, etc.)
    return None


class Pfeiffer_Block(Block):
    """
    A ViT block with a parallel Pfeiffer-style MLP adapter branch.
    Keeps the standard ViT residual ordering:
      x = x + DropPath( Attn( Norm1(x) ) )
      x = x + DropPath( MLP( Norm2(x) ) [+ Adapter] )
    """

    def __init__(self, adapter_config, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.adapter_config = adapter_config
        if adapter_config.STYLE != "Pfeiffer":
            raise ValueError("Only Pfeiffer adapter style is supported here.")

        red = max(1, dim // adapter_config.REDUCATION_FACTOR)
        self.adapter_downsample = nn.Linear(dim, red)
        self.adapter_act_fn = act_layer()
        self.adapter_upsample = nn.Linear(red, dim)

        # Zero-init so the adapter path is identity-safe at start
        nn.init.zeros_(self.adapter_downsample.weight)
        nn.init.zeros_(self.adapter_downsample.bias)
        nn.init.zeros_(self.adapter_upsample.weight)
        nn.init.zeros_(self.adapter_upsample.bias)

    def forward(self, x):
        # 1) MHSA residual
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # 2) MLP residual (+ Pfeiffer adapter in parallel)
        u = self.mlp(self.norm2(x))
        a = self.adapter_upsample(self.adapter_act_fn(self.adapter_downsample(self.norm2(x))))
        x = x + self.drop_path(u + a)
        return x
