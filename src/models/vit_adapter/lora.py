#!/usr/bin/env python3
"""LoRA layers for transformer PEFT baselines."""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap nn.Linear with a trainable low-rank residual branch."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # In LoRA experiments, the wrapped pretrained projection is frozen.
        for p in self.base.parameters():
            p.requires_grad = False

    @property
    def weight(self):  # compatibility for weight-loading code using .weight.copy_(...)
        return self.base.weight

    @property
    def bias(self):  # compatibility for weight-loading code using .bias.copy_(...)
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        z = F.linear(self.dropout(x), self.lora_A)
        z = F.linear(z, self.lora_B)
        return base_out + self.scaling * z


def normalize_targets(targets) -> set[str]:
    if targets is None:
        return {"query", "value"}
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    return {str(t).lower() for t in targets}


def apply_lora_to_attention(attn: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0, targets=None) -> None:
    """Replace selected Attention linear projections by LoRALinear in-place."""
    targets = normalize_targets(targets)
    name_map = {
        "q": "query",
        "query": "query",
        "k": "key",
        "key": "key",
        "v": "value",
        "value": "value",
        "o": "out",
        "out": "out",
        "proj": "out",
    }
    selected = {name_map[t] for t in targets if t in name_map}
    for name in selected:
        layer = getattr(attn, name)
        if not isinstance(layer, LoRALinear):
            setattr(attn, name, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))
