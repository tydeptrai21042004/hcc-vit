#!/usr/bin/env python3
"""Small transformer PEFT modules: AdaptFormer and SSF."""
from __future__ import annotations

import torch
import torch.nn as nn


class AdaptFormerAdapter(nn.Module):
    """
    Bottleneck adapter used by AdaptFormer-style ViT fine-tuning.

    The module is intentionally small and residual-scaled. It is inserted in the
    transformer block after the MLP branch, following the standard PEFT setting
    where the pretrained backbone is frozen and only adapter/head parameters are trained.
    """

    def __init__(self, hidden_size: int, reduction_factor: int = 16, scale: float = 1.0, dropout: float = 0.0):
        super().__init__()
        bottleneck = max(1, int(hidden_size) // max(1, int(reduction_factor)))
        self.adapter_down = nn.Linear(hidden_size, bottleneck)
        self.adapter_act = nn.GELU()
        self.adapter_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.adapter_up = nn.Linear(bottleneck, hidden_size)
        self.adapter_scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.adapter_down(x)
        z = self.adapter_act(z)
        z = self.adapter_dropout(z)
        z = self.adapter_up(z)
        return self.adapter_scale * z


class SSF(nn.Module):
    """Scale-and-shift feature modulation for SSF PEFT baseline."""

    def __init__(self, hidden_size: int, init_scale: float = 1.0, init_shift: float = 0.0):
        super().__init__()
        self.ssf_scale = nn.Parameter(torch.ones(hidden_size) * float(init_scale))
        self.ssf_shift = nn.Parameter(torch.ones(hidden_size) * float(init_shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.ssf_scale.view(*([1] * (x.ndim - 1)), -1) + self.ssf_shift.view(*([1] * (x.ndim - 1)), -1)
