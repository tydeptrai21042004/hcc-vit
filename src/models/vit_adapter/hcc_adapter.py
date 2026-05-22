# src/models/vit_adapter/hcc_adapter.py
"""
DT1D-Adapter / HCC token adapter for ViT.

This module updates the old HCC token adapter to match the current proposal:
  * finite weighted h-Hartley--cosine axial depthwise filtering;
  * optional multi-dilation branches;
  * static/global learnable softmax fusion over axis--dilation responses;
  * group-shared symmetric coefficients;
  * optional grouped pointwise bottleneck mixing;
  * scalar residual gate, normally initialized at 0 for identity-safe insertion.

Important terminology:
  The axis--scale gates are task-adaptive learnable parameters, but they are not
  input/sample-adaptive.  The older input-adaptive GAP-MLP router has been
  intentionally removed from this static-gate implementation.

Backward-compatible aliases and options are kept so old configs using NAME=HCC,
PER_CHANNEL, USE_PW, and H still run.
"""
from __future__ import annotations

import math
from math import gcd
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


DilationLike = Optional[Union[str, int, Sequence[int]]]


def _parse_dilations(dilations: DilationLike, fallback: int) -> Tuple[int, ...]:
    """Parse dilation specification into a tuple of positive unique integers."""
    if dilations is None:
        values = [int(fallback)]
    elif isinstance(dilations, int):
        values = [int(dilations)]
    elif isinstance(dilations, str):
        text = dilations.strip()
        if not text:
            values = [int(fallback)]
        else:
            text = text.replace(";", ",").replace(" ", ",")
            values = [int(v) for v in text.split(",") if v.strip()]
    else:
        values = [int(v) for v in dilations]

    clean = []
    for v in values:
        if v <= 0:
            raise ValueError(f"All dilations must be positive, got {values!r}")
        if v not in clean:
            clean.append(v)
    if not clean:
        clean = [int(fallback)]
    return tuple(clean)


class DT1DAdapter(nn.Module):
    """Spatial DT1D adapter on BCHW feature maps.

    Args:
        C: channel dimension.
        M: learnable symmetric coefficient radius. The learned sequence has
           M+1 coefficients per group, while the finite weighted-HCC kernel has
           effective support length 2M+3.
        h: legacy single dilation value.
        axis: one of "h", "w", or "hw".
        alpha_group: number of channels sharing one alpha coefficient group.
        dilations: optional int/list/string such as "1,2,4".
        scale_adaptive: if True, fuse all axis--dilation branches using global
           learnable softmax logits. Multiple dilations enable this automatically.
        separate_axis_kernels: if True in scale-adaptive mode, use different
           alpha groups per selected axis and dilation.
    """

    def __init__(
        self,
        C: int,
        M: int = 1,
        h: int = 1,
        axis: str = "hw",
        alpha_group: int = 16,
        tie_sym: bool = True,
        no_pw: bool = False,
        pw_ratio: int = 32,
        pw_groups: int = 4,
        use_bn: bool = False,
        residual_scale: float = 1.0,
        gate_init: float = 0.0,
        padding_mode: str = "reflect",
        dilations: DilationLike = None,
        scale_adaptive: bool = False,
        separate_axis_kernels: bool = True,
        gate_temperature: float = 1.0,
        input_adaptive_gate: bool = False,  # deprecated/ignored
        gate_reduction: int = 4,            # deprecated/ignored
        **legacy,
    ):
        super().__init__()

        if axis not in ("h", "w", "hw"):
            raise ValueError(f"axis must be one of 'h', 'w', 'hw', got {axis!r}")
        if padding_mode not in ("reflect", "replicate", "zeros", "constant"):
            raise ValueError(
                "padding_mode must be 'reflect', 'replicate', 'zeros', or 'constant', "
                f"got {padding_mode!r}"
            )

        # Legacy HCC config translation.
        if "per_channel" in legacy:
            per_channel = bool(legacy.pop("per_channel"))
            alpha_group = 1 if per_channel else alpha_group
        if "use_pw" in legacy:
            use_pw_legacy = bool(legacy.pop("use_pw"))
            no_pw = not use_pw_legacy
        if "hcc_dilations" in legacy and dilations is None:
            dilations = legacy.pop("hcc_dilations")
        legacy.pop("hcc_input_adaptive_gate", None)
        legacy.pop("hcc_gate_reduction", None)
        # Other unknown legacy keys are swallowed to preserve old launch scripts.

        self.C = int(C)
        self.M = int(M)
        self.h = int(h)
        self.axis = axis
        self.axis_names = tuple(a for a in ("h", "w") if a in axis)
        self.alpha_group = max(1, int(alpha_group))
        self.tie_sym = bool(tie_sym)
        self.no_pw = bool(no_pw)
        self.use_bn = bool(use_bn)
        self.residual_scale = float(residual_scale)
        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.dilations = _parse_dilations(dilations, fallback=self.h)
        self.scale_adaptive = bool(scale_adaptive or len(self.dilations) > 1)
        self.separate_axis_kernels = bool(separate_axis_kernels and self.scale_adaptive)
        self.gate_temperature = float(gate_temperature)
        self.input_adaptive_gate = False
        self.gate_reduction = max(1, int(gate_reduction))

        if self.C <= 0:
            raise ValueError(f"C must be positive, got {self.C}")
        if self.M < 0:
            raise ValueError(f"M must be non-negative, got {self.M}")
        if self.h <= 0:
            raise ValueError(f"h/dilation must be positive, got {self.h}")
        if self.gate_temperature <= 0:
            raise ValueError(f"gate_temperature must be positive, got {self.gate_temperature}")

        self.num_alpha_groups = math.ceil(self.C / self.alpha_group)
        self.num_axes = len(self.axis_names)
        self.num_scales = len(self.dilations)
        self.num_alpha_axes = self.num_axes if self.separate_axis_kernels else 1
        ncoef = self.M + 1

        self.alpha = nn.Parameter(
            torch.zeros(self.num_alpha_axes, self.num_scales, self.num_alpha_groups, ncoef)
        )
        with torch.no_grad():
            # This makes the axial branch nonzero, but identity safety comes from gate_init=0.
            self.alpha[..., 0].fill_(1.0)

        if self.scale_adaptive:
            self.axis_scale_logits = nn.Parameter(torch.zeros(self.num_axes, self.num_scales))
        else:
            self.register_parameter("axis_scale_logits", None)
        # No input-adaptive router in the proposal method.
        self.axis_scale_router = None

        if not self.no_pw:
            hidden = max(1, self.C // max(1, int(pw_ratio)))
            groups = max(1, int(pw_groups))
            groups = min(groups, self.C, hidden)
            groups = gcd(groups, self.C)
            groups = gcd(groups, hidden) or 1
            self.pw_groups = groups
            self.pw = nn.Sequential(
                nn.Conv2d(self.C, hidden, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(hidden) if self.use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, self.C, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(self.C) if self.use_bn else nn.Identity(),
            )
        else:
            self.pw_groups = 1
            self.pw = nn.Identity()

        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def extra_repr(self) -> str:
        return (
            f"C={self.C}, M={self.M}, dilations={self.dilations}, axis={self.axis}, "
            f"scale_adaptive={self.scale_adaptive}, static_axis_scale_gate=True, "
            f"separate_axis_kernels={self.separate_axis_kernels}, "
            f"alpha_group={self.alpha_group}, G={self.num_alpha_groups}, "
            f"no_pw={self.no_pw}, gate={float(self.gate.detach().cpu()):.4g}"
        )

    def parameter_count_breakdown(self) -> Dict[str, int]:
        axial = self.alpha.numel() + self.gate.numel()
        axis_scale = 0 if self.axis_scale_logits is None else self.axis_scale_logits.numel()
        pw = sum(p.numel() for p in self.pw.parameters())
        return {
            "axial_alpha_and_gate": int(axial),
            "axis_scale_logits": int(axis_scale),
            "pointwise": int(pw),
            "total": int(axial + axis_scale + pw),
        }

    def axis_scale_weights(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Return detached static softmax weights with shape (num_axes, num_scales)."""
        if self.axis_scale_logits is None:
            return None
        logits = self.axis_scale_logits.detach() / self.gate_temperature
        return torch.softmax(logits.reshape(-1), dim=0).reshape(self.num_axes, self.num_scales)

    def _compute_axis_scale_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.axis_scale_logits is None:
            raise RuntimeError("Axis--scale weights are only defined in scale_adaptive mode.")
        logits = self.axis_scale_logits.to(device=device, dtype=dtype) / self.gate_temperature
        weights = torch.softmax(logits.reshape(-1), dim=0)
        return weights.reshape(self.num_axes, self.num_scales)

    def _expand_group_kernel_to_channels(self, wg: torch.Tensor) -> torch.Tensor:
        chunks = []
        remaining = self.C
        for g in range(self.num_alpha_groups):
            rep = min(self.alpha_group, remaining)
            chunks.append(wg[g].unsqueeze(0).repeat(rep, 1))
            remaining -= rep
        w = torch.cat(chunks, dim=0)
        if w.shape[0] != self.C:
            raise RuntimeError(f"Internal error: built {w.shape[0]} channel kernels for C={self.C}")
        return w.unsqueeze(1)

    def _build_weighted_hcc_kernel_1d(
        self,
        axis_idx: int,
        scale_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build finite weighted h-Hartley--cosine axial kernels.

        The learned sequence f[-M],...,f[M] is symmetric and group-shared:
        f[0]=alpha_0, f[+m]=f[-m]=alpha_m.  The four weighted-HCC shifts induce
        an ordinary 1D depthwise kernel supported on [-(M+1), ..., M+1], so the
        effective kernel length is 2M+3.
        """
        K_eff = 2 * self.M + 3
        center = self.M + 1
        alpha_axis_idx = axis_idx if self.separate_axis_kernels else 0
        alpha = self.alpha[alpha_axis_idx, scale_idx].to(device=device, dtype=dtype)

        wg = torch.zeros(self.num_alpha_groups, K_eff, device=device, dtype=dtype)
        for m in range(-self.M, self.M + 1):
            val = alpha[:, abs(m)]
            for r in (-(m + 1), -(m - 1), (m + 1), (m - 1)):
                wg[:, center + r] += 0.5 * val

        denom = wg.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        wg = wg / denom
        return self._expand_group_kernel_to_channels(wg)

    def _pad(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return x
        if self.padding_mode == "constant":
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)
        mode = self.padding_mode
        if mode == "reflect":
            H, W = x.shape[-2], x.shape[-1]
            # reflect requires pad < dimension; use replicate for tiny grids.
            if (pad_h >= H and pad_h > 0) or (pad_w >= W and pad_w > 0):
                mode = "replicate"
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=mode)

    def _conv_axis(self, x: torch.Tensor, axis_name: str, w1d: torch.Tensor, dilation: int) -> torch.Tensor:
        K = int(w1d.shape[-1])
        radius = K // 2
        pad = radius * dilation
        if axis_name == "h":
            weight = w1d.view(self.C, 1, K, 1)
            x_pad = self._pad(x, pad_h=pad, pad_w=0)
            return F.conv2d(x_pad, weight, stride=1, padding=0, dilation=(dilation, 1), groups=self.C)
        if axis_name == "w":
            weight = w1d.view(self.C, 1, 1, K)
            x_pad = self._pad(x, pad_h=0, pad_w=pad)
            return F.conv2d(x_pad, weight, stride=1, padding=0, dilation=(1, dilation), groups=self.C)
        raise ValueError(f"Unknown axis_name={axis_name!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DT1DAdapter expects BCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.C:
            raise ValueError(f"Channel mismatch: adapter C={self.C}, input C={x.shape[1]}")

        if self.scale_adaptive:
            weights = self._compute_axis_scale_weights(x.device, x.dtype)
            y = torch.zeros_like(x)
            for ai, axis_name in enumerate(self.axis_names):
                for si, dilation in enumerate(self.dilations):
                    w1d = self._build_weighted_hcc_kernel_1d(ai, si, x.device, x.dtype)
                    yi = self._conv_axis(x, axis_name, w1d, dilation)
                    y = y + weights[ai, si] * yi
        else:
            y = None
            n_axes = 0
            scale_idx = 0
            dilation = self.dilations[0]
            for ai, axis_name in enumerate(self.axis_names):
                w1d = self._build_weighted_hcc_kernel_1d(ai, scale_idx, x.device, x.dtype)
                yi = self._conv_axis(x, axis_name, w1d, dilation)
                y = yi if y is None else y + yi
                n_axes += 1
            if y is None:
                y = x
                n_axes = 1
            y = y / float(max(1, n_axes))

        y = self.pw(y)
        return x + self.residual_scale * self.gate * y


class HCCTokenAdapter(nn.Module):
    """Apply DT1DAdapter to ViT patch tokens and pass the class token through.

    Input shape is (B, N, D), where N = 1 + H*W and token 0 is the class token.
    Patch tokens are reshaped to (B, D, H, W), processed by DT1DAdapter, and
    reshaped back to token form.
    """

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
        dilations: DilationLike = None,
        scale_adaptive: bool = False,
        separate_axis_kernels: bool = True,
        gate_temperature: float = 1.0,
        input_adaptive_gate: bool = False,
        gate_reduction: int = 4,
        per_channel: Optional[bool] = None,
        use_pw: Optional[bool] = None,
        **legacy,
    ):
        super().__init__()
        self.D = int(embed_dim)
        gh, gw = int(grid_size[0]), int(grid_size[1])
        if gh <= 0 or gw <= 0:
            raise ValueError(f"invalid grid_size: {grid_size}")
        self.gh, self.gw = gh, gw

        # Legacy wrappers passed per_channel/use_pw as explicit kwargs rather than inside **legacy.
        if per_channel is not None:
            legacy["per_channel"] = per_channel
        if use_pw is not None:
            legacy["use_pw"] = use_pw

        self.hcc = DT1DAdapter(
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
            dilations=dilations,
            scale_adaptive=scale_adaptive,
            separate_axis_kernels=separate_axis_kernels,
            gate_temperature=gate_temperature,
            input_adaptive_gate=input_adaptive_gate,
            gate_reduction=gate_reduction,
            **legacy,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"HCCTokenAdapter expects BND tokens, got shape {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.D:
            raise ValueError(f"embed_dim mismatch: got {D}, expected {self.D}")
        expected = 1 + self.gh * self.gw
        if N != expected:
            raise ValueError(f"N={N} but grid {self.gh}x{self.gw} implies {expected}")

        cls_tok = x[:, :1, :]
        patches = x[:, 1:, :]
        fmap = patches.transpose(1, 2).reshape(B, D, self.gh, self.gw)
        fmap_out = self.hcc(fmap)
        patches_out = fmap_out.reshape(B, D, self.gh * self.gw).transpose(1, 2)
        return torch.cat([cls_tok, patches_out], dim=1)

    def axis_scale_weights(self) -> Optional[torch.Tensor]:
        return self.hcc.axis_scale_weights()

    def parameter_count_breakdown(self) -> Dict[str, int]:
        return self.hcc.parameter_count_breakdown()


# Backward-compatible aliases used by old code/configs.
HCCAdapter = DT1DAdapter
H1D_DT_Adapter = DT1DAdapter
OneDDTAdapter = DT1DAdapter
DT1DTokenAdapter = HCCTokenAdapter
