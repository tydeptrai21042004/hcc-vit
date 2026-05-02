#!/usr/bin/env python3
"""
DT1D/HCC token adapter for Vision Transformer experiments.

This implementation mirrors the corrected 1D-DT adapter behavior:
  * axial depthwise 1D filtering on H/W patch-token grids;
  * group-shared symmetric coefficients with ceil(C / channels_per_group);
  * safe residual gate initialization, default 0.0;
  * optional grouped point-wise bottleneck;
  * axis='hw' averages the H and W responses instead of summing them.
"""
from __future__ import annotations

import math
from math import gcd
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HCCAdapter(nn.Module):
    """
    Axial depthwise DT1D/HCC adapter over feature maps.

    Args:
        C: Number of channels / ViT embedding dimension.
        M: Number of side taps. Kernel size is 2*M + 1.
        h: Dilation factor.
        axis: 'h', 'w', or 'hw'.
        alpha_group: Number of channels sharing one alpha vector.
        tie_sym: Keep left/right taps tied. This implementation is symmetric.
        no_pw: If True, disables point-wise bottleneck mixing.
        pw_ratio: Reduction ratio for optional point-wise bottleneck.
        pw_groups: Maximum grouped-conv groups for point-wise bottleneck.
        gate_init: Residual gate initialization. Use 0.0 for identity-safe PEFT.
        per_channel/use_pw: Legacy compatibility with the original HCC config.
    """

    def __init__(
        self,
        C: int,
        M: int = 1,
        h: int = 1,
        axis: str = "hw",
        alpha_group: int = 16,
        tie_sym: bool = True,
        no_pw: bool = True,
        pw_ratio: int = 32,
        pw_groups: int = 4,
        use_bn: bool = False,
        residual_scale: float = 1.0,
        gate_init: float = 0.0,
        padding_mode: str = "reflect",
        per_channel: Optional[bool] = None,
        use_pw: Optional[bool] = None,
        **legacy,
    ) -> None:
        super().__init__()
        if axis not in {"h", "w", "hw"}:
            raise ValueError(f"axis must be 'h', 'w', or 'hw', got {axis!r}")
        if C <= 0:
            raise ValueError("C must be positive")
        if M < 0:
            raise ValueError("M must be non-negative")
        if h <= 0:
            raise ValueError("h must be positive")

        if per_channel is not None:
            alpha_group = 1 if bool(per_channel) else int(alpha_group)
        if use_pw is not None:
            no_pw = not bool(use_pw)

        self.C = int(C)
        self.M = int(M)
        self.h = int(h)
        self.axis = str(axis)
        self.alpha_group = max(1, int(alpha_group))
        self.tie_sym = bool(tie_sym)
        self.no_pw = bool(no_pw)
        self.use_bn = bool(use_bn)
        self.residual_scale = float(residual_scale)
        self.padding_mode = str(padding_mode).lower()

        # Correct 1D-DT grouping: alpha_group means channels per shared filter.
        # ceil handles remainder channels instead of dropping them.
        self.num_alpha_groups = int(math.ceil(self.C / self.alpha_group))
        ncoef = self.M + 1
        self.alpha = nn.Parameter(torch.zeros(self.num_alpha_groups, ncoef))
        with torch.no_grad():
            self.alpha[:, 0].fill_(1.0)

        if not self.no_pw:
            hidden = max(1, self.C // max(1, int(pw_ratio)))
            g = max(1, min(int(pw_groups), self.C, hidden))
            g = gcd(g, self.C)
            g = gcd(g, hidden) or 1
            self.pw_groups = int(g)
            self.pw = nn.Sequential(
                nn.Conv2d(self.C, hidden, kernel_size=1, groups=self.pw_groups, bias=False),
                nn.BatchNorm2d(hidden) if self.use_bn else nn.Identity(),
                nn.GELU(),
                nn.Conv2d(hidden, self.C, kernel_size=1, groups=self.pw_groups, bias=False),
                nn.BatchNorm2d(self.C) if self.use_bn else nn.Identity(),
            )
        else:
            self.pw_groups = 1
            self.pw = nn.Identity()

        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def _build_even_kernel_1d(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return depthwise conv weight of shape (C, 1, 2*M+1)."""
        K = 2 * self.M + 1
        center = self.M
        wg = torch.zeros(self.num_alpha_groups, K, device=device, dtype=dtype)
        wg[:, center] = self.alpha[:, 0].to(dtype=dtype)
        for m in range(1, self.M + 1):
            val = self.alpha[:, m].to(dtype=dtype)
            wg[:, center - m] = val
            wg[:, center + m] = val  # symmetric/tied DT1D kernel

        # Stable per-group normalization. Keeps identity kernel unchanged at init.
        denom = wg.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        wg = wg / denom

        chunks = []
        for gi in range(self.num_alpha_groups):
            start = gi * self.alpha_group
            end = min((gi + 1) * self.alpha_group, self.C)
            n = end - start
            if n > 0:
                chunks.append(wg[gi].unsqueeze(0).repeat(n, 1))
        w = torch.cat(chunks, dim=0)
        if w.shape[0] != self.C:
            raise RuntimeError(f"internal grouping error: built {w.shape[0]} kernels for C={self.C}")
        return w.unsqueeze(1)

    @staticmethod
    def _can_reflect(size: int, pad: int) -> bool:
        # PyTorch reflect padding requires pad < input dimension.
        return pad == 0 or pad < size

    def _pad(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        pads = (pad_w, pad_w, pad_h, pad_h)
        if pad_h == 0 and pad_w == 0:
            return x

        mode = self.padding_mode
        if mode == "reflect" and not (
            self._can_reflect(x.shape[-2], pad_h) and self._can_reflect(x.shape[-1], pad_w)
        ):
            mode = "replicate"

        if mode in {"reflect", "replicate", "circular"}:
            return F.pad(x, pads, mode=mode)
        return F.pad(x, pads, mode="constant", value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)."""
        if x.ndim != 4:
            raise ValueError(f"HCCAdapter expects BCHW tensor, got shape {tuple(x.shape)}")
        B, C, H, W = x.shape
        if C != self.C:
            raise ValueError(f"channel mismatch: got {C}, expected {self.C}")

        w1d = self._build_even_kernel_1d(x.device, x.dtype)
        K = 2 * self.M + 1
        y = torch.zeros_like(x)
        n_axes = 0

        if "h" in self.axis:
            wh = w1d.view(self.C, 1, K, 1)
            xh = self._pad(x, pad_h=self.M * self.h, pad_w=0)
            y = y + F.conv2d(xh, wh, stride=1, padding=0, dilation=(self.h, 1), groups=self.C)
            n_axes += 1

        if "w" in self.axis:
            ww = w1d.view(self.C, 1, 1, K)
            xw = self._pad(x, pad_h=0, pad_w=self.M * self.h)
            y = y + F.conv2d(xw, ww, stride=1, padding=0, dilation=(1, self.h), groups=self.C)
            n_axes += 1

        if n_axes > 1:
            y = y / float(n_axes)
        y = self.pw(y)
        return x + self.residual_scale * self.gate * y


class HCCTokenAdapter(nn.Module):
    """Apply HCCAdapter to ViT patch tokens while preserving the class token."""

    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int],
        M: int = 1,
        h: int = 1,
        axis: str = "hw",
        alpha_group: int = 16,
        tie_sym: bool = True,
        no_pw: bool = True,
        pw_ratio: int = 32,
        pw_groups: int = 4,
        use_bn: bool = False,
        residual_scale: float = 1.0,
        gate_init: float = 0.0,
        padding_mode: str = "reflect",
        per_channel: Optional[bool] = None,
        use_pw: Optional[bool] = None,
        **legacy,
    ) -> None:
        super().__init__()
        self.D = int(embed_dim)
        self.gh, self.gw = int(grid_size[0]), int(grid_size[1])
        if self.gh <= 0 or self.gw <= 0:
            raise ValueError(f"invalid grid_size={grid_size}")
        self.hcc = HCCAdapter(
            C=self.D,
            M=M,
            h=h,
            axis=axis,
            alpha_group=alpha_group,
            tie_sym=tie_sym,
            no_pw=no_pw,
            pw_ratio=pw_ratio,
            pw_groups=pw_groups,
            use_bn=use_bn,
            residual_scale=residual_scale,
            gate_init=gate_init,
            padding_mode=padding_mode,
            per_channel=per_channel,
            use_pw=use_pw,
            **legacy,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"HCCTokenAdapter expects BND tensor, got shape {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.D:
            raise ValueError(f"embed_dim mismatch: got {D}, expected {self.D}")
        expected = 1 + self.gh * self.gw
        if N != expected:
            raise ValueError(f"N={N}, but grid {self.gh}x{self.gw} implies {expected} tokens")

        cls_tok = x[:, :1, :]
        patches = x[:, 1:, :]
        fmap = patches.transpose(1, 2).reshape(B, D, self.gh, self.gw)
        fmap = self.hcc(fmap)
        patches = fmap.reshape(B, D, self.gh * self.gw).transpose(1, 2)
        return torch.cat([cls_tok, patches], dim=1)
