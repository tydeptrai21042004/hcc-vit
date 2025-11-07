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
        return HCCTokenAdapter(
            embed_dim=embed_dim,
            grid_size=grid_size,                # (H, W) of patch grid
            M=cfg.ADAPTER.HCC.M,
            h=cfg.ADAPTER.HCC.H,
            axis=cfg.ADAPTER.HCC.AXIS,
            per_channel=cfg.ADAPTER.HCC.PER_CHANNEL,
            tie_sym=cfg.ADAPTER.HCC.TIE_SYM,
            use_pw=cfg.ADAPTER.HCC.USE_PW,
            pw_ratio=cfg.ADAPTER.HCC.PW_RATIO,
            use_bn=cfg.ADAPTER.HCC.USE_BN,
            residual_scale=cfg.ADAPTER.HCC.RESIDUAL_SCALE,
            gate_init=cfg.ADAPTER.HCC.GATE_INIT,
            padding_mode=cfg.ADAPTER.HCC.PADDING,
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
