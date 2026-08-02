#!/usr/bin/env python3
"""ViT blocks with Pfeiffer, HCC-DT1D, and HOSQ-DT1D adapters."""
from __future__ import annotations

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

from ...utils import logging as vlogging
from .hcc_adapter import HCCTokenAdapter, HOSQTokenAdapter

logger = vlogging.get_logger("visual_prompt")


def _adapter_name(adapter_config) -> str:
    """Resolve old STYLE-only configs and new explicit NAME configs."""
    name = str(getattr(adapter_config, "NAME", "")).strip().lower()
    if name in ("", "none", "null"):
        style = str(getattr(adapter_config, "STYLE", "Pfeiffer")).strip().lower()
        if style == "pfeiffer":
            return "pfeiffer"
    return name


def build_adapter(name: str, embed_dim: int, grid_size, adapter_config):
    """Construct a token-space HCC or HOSQ adapter."""
    name = str(name).lower()
    if grid_size is None:
        raise ValueError(f"grid_size is required for token adapter {name!r}")

    if name == "hcc":
        hcc = adapter_config.HCC
        return HCCTokenAdapter(
            embed_dim=embed_dim,
            grid_size=grid_size,
            M=getattr(hcc, "M", 1),
            h=getattr(hcc, "H", 1),
            axis=getattr(hcc, "AXIS", "hw"),
            alpha_group=getattr(hcc, "ALPHA_GROUP", 16),
            per_channel=getattr(hcc, "PER_CHANNEL", None),
            tie_sym=getattr(hcc, "TIE_SYM", True),
            no_pw=getattr(hcc, "NO_PW", not getattr(hcc, "USE_PW", False)),
            use_pw=getattr(hcc, "USE_PW", None),
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

    if name == "hosq":
        hosq = adapter_config.HOSQ
        return HOSQTokenAdapter(
            embed_dim=embed_dim,
            grid_size=grid_size,
            num_prefix_tokens=getattr(hosq, "NUM_PREFIX_TOKENS", 1),
            axis=getattr(hosq, "AXIS", "hw"),
            coarse_group=getattr(hosq, "COARSE_GROUP", 32),
            subgroup_size=getattr(hosq, "SUBGROUP_SIZE", 8),
            rank4=getattr(hosq, "RANK4", 1),
            rank8=getattr(hosq, "RANK8", 2),
            no_pw=getattr(hosq, "NO_PW", True),
            pw_ratio=getattr(hosq, "PW_RATIO", 32),
            pw_groups=getattr(hosq, "PW_GROUPS", 4),
            use_bn=getattr(hosq, "USE_BN", False),
            residual_scale=getattr(hosq, "RESIDUAL_SCALE", 1.0),
            gate_init=getattr(hosq, "GATE_INIT", 0.01),
            padding_mode=getattr(hosq, "PADDING", "reflect"),
            strict_padding=getattr(hosq, "STRICT_PADDING", True),
        )

    raise ValueError(f"Unsupported token adapter name {name!r}")


class Pfeiffer_Block(Block):
    """A timm ViT block supporting three adapter realizations.

    The class name is retained for checkpoint and import compatibility.  In
    HCC/HOSQ mode, the token adapter is applied to the MLP output at the same
    insertion point used by the supervised ViT implementation.
    """

    def __init__(
        self,
        adapter_config,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        grid_size=None,
    ):
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
        self.adapter_name = _adapter_name(adapter_config)
        self.token_adapter = None

        if self.adapter_name == "pfeiffer":
            reduction_factor = max(1, int(adapter_config.REDUCATION_FACTOR))
            reduced_dim = max(1, dim // reduction_factor)
            self.adapter_downsample = nn.Linear(dim, reduced_dim)
            self.adapter_act_fn = act_layer()
            self.adapter_upsample = nn.Linear(reduced_dim, dim)
            nn.init.zeros_(self.adapter_downsample.weight)
            nn.init.zeros_(self.adapter_downsample.bias)
            nn.init.zeros_(self.adapter_upsample.weight)
            nn.init.zeros_(self.adapter_upsample.bias)
        elif self.adapter_name in ("hcc", "hosq"):
            self.token_adapter = build_adapter(
                self.adapter_name, dim, grid_size, adapter_config
            )
        else:
            raise ValueError(
                f"Adapter NAME={getattr(adapter_config, 'NAME', None)!r} is not supported"
            )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        normalized = self.norm2(x)
        mlp_output = self.mlp(normalized)

        if self.adapter_name == "pfeiffer":
            adapter_output = self.adapter_upsample(
                self.adapter_act_fn(self.adapter_downsample(normalized))
            )
            residual = mlp_output + adapter_output
        else:
            # HCCTokenAdapter/HOSQTokenAdapter contain their own residual gate,
            # so this returns an adapted version of the MLP output.
            residual = self.token_adapter(mlp_output)

        return x + self.drop_path(residual)
