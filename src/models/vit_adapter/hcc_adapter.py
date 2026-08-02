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


class HOSQDT1DAdapter(nn.Module):
    """Hierarchically Orthogonal Spectral Quotient DT1D adapter.

    The adapter learns a coarse symmetric MLQ8 kernel on channel groups of
    ``coarse_group`` channels and adds a low-rank subgroup correction.  The
    subgroup correction uses fixed zero-mean orthonormal channel contrasts and
    zero-DC spatial atoms at offsets 4 and 8.  One ordinary depthwise
    convolution is executed per enabled spatial axis.

    The final residual map is

        output = x + residual_scale * gate * pointwise(T_hosq(x)).

    ``no_pw=True`` is the proposed HOSQ setting.  Pointwise mixing remains
    available only so a controlled with/without-pointwise ablation can be run.
    """

    _OFFSETS: Tuple[int, ...] = (0, 1, 2, 4, 8)

    def __init__(
        self,
        C: int,
        axis: str = "hw",
        coarse_group: int = 32,
        subgroup_size: int = 8,
        rank4: int = 1,
        rank8: int = 2,
        no_pw: bool = True,
        pw_ratio: int = 32,
        pw_groups: int = 4,
        use_bn: bool = False,
        residual_scale: float = 1.0,
        gate_init: float = 0.01,
        padding_mode: str = "reflect",
        strict_padding: bool = True,
    ):
        super().__init__()

        if axis not in ("h", "w", "hw"):
            raise ValueError(f"axis must be one of 'h', 'w', 'hw', got {axis!r}")
        if padding_mode not in ("reflect", "replicate", "zeros", "constant"):
            raise ValueError(
                "padding_mode must be 'reflect', 'replicate', 'zeros', or 'constant', "
                f"got {padding_mode!r}"
            )

        self.C = int(C)
        self.axis = str(axis)
        self.axis_names = tuple(a for a in ("h", "w") if a in self.axis)
        self.num_axes = len(self.axis_names)
        self.coarse_group = int(coarse_group)
        self.subgroup_size = int(subgroup_size)
        self.rank4_requested = int(rank4)
        self.rank8_requested = int(rank8)
        self.no_pw = bool(no_pw)
        self.use_bn = bool(use_bn)
        self.residual_scale = float(residual_scale)
        self.padding_mode = "constant" if padding_mode == "zeros" else str(padding_mode)
        self.strict_padding = bool(strict_padding)

        if self.C <= 0:
            raise ValueError(f"C must be positive, got {self.C}")
        if self.coarse_group <= 0:
            raise ValueError(f"coarse_group must be positive, got {self.coarse_group}")
        if self.subgroup_size <= 0:
            raise ValueError(f"subgroup_size must be positive, got {self.subgroup_size}")
        if self.subgroup_size > self.coarse_group:
            raise ValueError("subgroup_size cannot exceed coarse_group")
        if self.coarse_group % self.subgroup_size != 0:
            raise ValueError("coarse_group must be divisible by subgroup_size")
        if self.rank4_requested < 0 or self.rank8_requested < 0:
            raise ValueError("rank4 and rank8 must be non-negative")

        self.num_coarse_groups = math.ceil(self.C / self.coarse_group)
        max_subgroups = max(
            1, math.ceil(min(self.C, self.coarse_group) / self.subgroup_size)
        )
        max_contrasts = max(0, max_subgroups - 1)
        self.rank4 = min(self.rank4_requested, max_contrasts)
        self.rank8 = min(self.rank8_requested, max_contrasts)

        # Coarse observable quotient coordinates ordered as beta_0, beta_1,
        # beta_2, beta_4, beta_8 for every axis and Group-32 channel group.
        self.quotient_beta = nn.Parameter(
            torch.zeros(self.num_axes, self.num_coarse_groups, len(self._OFFSETS))
        )

        basis, channel_group, channel_subgroup, subgroup_counts = \
            self._make_hosq_index_buffers()
        detail4_group, detail4_mode = self._make_detail_coordinate_map(
            subgroup_counts, self.rank4
        )
        detail8_group, detail8_mode = self._make_detail_coordinate_map(
            subgroup_counts, self.rank8
        )

        # Only mathematically valid coordinates are allocated.  In particular,
        # an incomplete final coarse group does not create unused detail weights.
        self.detail4 = nn.Parameter(
            torch.zeros(self.num_axes, int(detail4_group.numel()))
        )
        self.detail8 = nn.Parameter(
            torch.zeros(self.num_axes, int(detail8_group.numel()))
        )

        self.register_buffer("hosq_basis", basis, persistent=True)
        self.register_buffer("channel_group", channel_group, persistent=True)
        self.register_buffer("channel_subgroup", channel_subgroup, persistent=True)
        self.register_buffer("subgroup_counts", subgroup_counts, persistent=True)
        self.register_buffer("detail4_group", detail4_group, persistent=True)
        self.register_buffer("detail4_mode", detail4_mode, persistent=True)
        self.register_buffer("detail8_group", detail8_group, persistent=True)
        self.register_buffer("detail8_mode", detail8_mode, persistent=True)

        # Equal-route initialization inherited from the three-dilation DT1D
        # operator.  Joint L1 mass is one before the residual gate is applied.
        with torch.no_grad():
            init_side = 1.0 / float(2 * self.num_axes * 3)
            self.quotient_beta[..., 1:4].fill_(init_side)

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
            f"C={self.C}, axis={self.axis}, coarse_group={self.coarse_group}, "
            f"subgroup_size={self.subgroup_size}, ranks=({self.rank4},{self.rank8}), "
            f"groups={self.num_coarse_groups}, no_pw={self.no_pw}, "
            f"padding={self.padding_mode}, gate={float(self.gate.detach().cpu()):.4g}"
        )

    @staticmethod
    def _orthogonal_subgroup_basis(
        n: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return a zero-mean orthonormal contrast basis on ``n`` subgroups."""
        dtype = dtype or torch.float32
        if n <= 1:
            return torch.zeros(n, 0, device=device, dtype=dtype)
        if n == 4:
            return torch.tensor(
                [
                    [0.5, 1.0 / math.sqrt(2.0), 0.0],
                    [0.5, -1.0 / math.sqrt(2.0), 0.0],
                    [-0.5, 0.0, 1.0 / math.sqrt(2.0)],
                    [-0.5, 0.0, -1.0 / math.sqrt(2.0)],
                ],
                device=device,
                dtype=dtype,
            )

        # Canonical Helmert contrasts for incomplete remainder groups.
        basis = torch.zeros(n, n - 1, device=device, dtype=dtype)
        for k in range(1, n):
            denom = math.sqrt(float(k * (k + 1)))
            basis[:k, k - 1] = 1.0 / denom
            basis[k, k - 1] = -float(k) / denom
        return basis

    def _make_hosq_index_buffers(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_subgroups = max(1, math.ceil(self.coarse_group / self.subgroup_size))
        max_rank = max(0, max_subgroups - 1)
        basis = torch.zeros(self.num_coarse_groups, max_subgroups, max_rank)
        channel_group = torch.empty(self.C, dtype=torch.long)
        channel_subgroup = torch.empty(self.C, dtype=torch.long)
        subgroup_counts = torch.empty(self.num_coarse_groups, dtype=torch.long)

        start = 0
        for group_idx in range(self.num_coarse_groups):
            group_channels = min(self.coarse_group, self.C - start)
            n_subgroups = max(1, math.ceil(group_channels / self.subgroup_size))
            subgroup_counts[group_idx] = n_subgroups
            local_basis = self._orthogonal_subgroup_basis(n_subgroups)
            active_rank = min(max_rank, local_basis.shape[1])
            if active_rank:
                basis[group_idx, :n_subgroups, :active_rank] = local_basis[:, :active_rank]
            for local_channel in range(group_channels):
                channel_group[start + local_channel] = group_idx
                channel_subgroup[start + local_channel] = min(
                    local_channel // self.subgroup_size, n_subgroups - 1
                )
            start += group_channels
        return basis, channel_group, channel_subgroup, subgroup_counts

    @staticmethod
    def _make_detail_coordinate_map(
        subgroup_counts: torch.Tensor,
        requested_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        groups = []
        modes = []
        for group_idx, count in enumerate(subgroup_counts.tolist()):
            active = min(int(requested_rank), max(0, int(count) - 1))
            for mode in range(active):
                groups.append(group_idx)
                modes.append(mode)
        return torch.tensor(groups, dtype=torch.long), torch.tensor(modes, dtype=torch.long)

    def _expand_channel_detail(
        self,
        theta: torch.Tensor,
        coordinate_groups: torch.Tensor,
        coordinate_modes: torch.Tensor,
        group_index: torch.Tensor,
        subgroup_index: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        if theta.shape[1] == 0:
            return torch.zeros(
                self.num_axes, self.C, device=theta.device, dtype=theta.dtype
            )
        coordinate_groups = coordinate_groups.to(theta.device)
        coordinate_modes = coordinate_modes.to(theta.device)
        selected_group_basis = basis[coordinate_groups]
        gather_index = coordinate_modes.view(-1, 1, 1).expand(
            -1, selected_group_basis.shape[1], 1
        )
        selected_basis = selected_group_basis.gather(2, gather_index).squeeze(2)
        contributions = theta.unsqueeze(-1) * selected_basis.unsqueeze(0)
        per_group = torch.zeros(
            self.num_axes,
            self.num_coarse_groups,
            basis.shape[1],
            device=theta.device,
            dtype=theta.dtype,
        )
        per_group.index_add_(1, coordinate_groups, contributions)
        return per_group[:, group_index, subgroup_index]

    def build_normalized_kernels(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build jointly projected kernels with shape ``(A,C,1,17)``."""
        group_index = self.channel_group.to(device=device)
        subgroup_index = self.channel_subgroup.to(device=device)
        beta = self.quotient_beta.to(device=device, dtype=dtype)[:, group_index, :]
        kernel = torch.zeros(self.num_axes, self.C, 17, device=device, dtype=dtype)
        center = 8
        kernel[..., center] = beta[..., 0]
        for coefficient_index, offset in enumerate((1, 2, 4, 8), start=1):
            kernel[..., center - offset] = beta[..., coefficient_index]
            kernel[..., center + offset] = beta[..., coefficient_index]

        basis = self.hosq_basis.to(device=device, dtype=dtype)
        if self.detail4.shape[1] > 0:
            detail4 = self._expand_channel_detail(
                self.detail4.to(device=device, dtype=dtype),
                self.detail4_group,
                self.detail4_mode,
                group_index,
                subgroup_index,
                basis,
            )
            kernel[..., center - 4] += detail4
            kernel[..., center + 4] += detail4
            kernel[..., center] -= 2.0 * detail4
        if self.detail8.shape[1] > 0:
            detail8 = self._expand_channel_detail(
                self.detail8.to(device=device, dtype=dtype),
                self.detail8_group,
                self.detail8_mode,
                group_index,
                subgroup_index,
                basis,
            )
            kernel[..., center - 8] += detail8
            kernel[..., center + 8] += detail8
            kernel[..., center] -= 2.0 * detail8

        # Joint projection across both axes gives
        # sum_a ||k_{a,c}||_1 <= 1 for every channel c.
        joint_l1 = kernel.abs().sum(dim=-1).sum(dim=0)
        scale = torch.maximum(joint_l1, torch.ones_like(joint_l1)).view(1, self.C, 1)
        return (kernel / scale).unsqueeze(2)

    def parameter_count_breakdown(self) -> Dict[str, int]:
        coarse = self.quotient_beta.numel()
        detail4 = self.detail4.numel()
        detail8 = self.detail8.numel()
        pointwise = sum(parameter.numel() for parameter in self.pw.parameters())
        total = coarse + detail4 + detail8 + self.gate.numel() + pointwise
        return {
            "coarse_quotient": int(coarse),
            "detail_offset4": int(detail4),
            "detail_offset8": int(detail8),
            "residual_gate": int(self.gate.numel()),
            "pointwise": int(pointwise),
            "total": int(total),
        }

    def mathematical_invariants(self) -> Dict[str, float]:
        """Return deterministic numerical checks of the fixed HOSQ structure."""
        max_mean_error = 0.0
        max_gram_error = 0.0
        for group_idx, count in enumerate(self.subgroup_counts.tolist()):
            if count <= 1:
                continue
            basis = self.hosq_basis[group_idx, :count, :count - 1]
            mean_error = basis.sum(dim=0).abs().max().item()
            gram = basis.transpose(0, 1) @ basis
            gram_error = (gram - torch.eye(count - 1, dtype=gram.dtype)).abs().max().item()
            max_mean_error = max(max_mean_error, mean_error)
            max_gram_error = max(max_gram_error, gram_error)
        kernels = self.build_normalized_kernels(torch.device("cpu"), torch.float32)
        joint_l1 = kernels.squeeze(2).abs().sum(dim=-1).sum(dim=0).max().item()
        return {
            "basis_zero_mean_error": float(max_mean_error),
            "basis_gram_error": float(max_gram_error),
            "maximum_joint_axis_l1": float(joint_l1),
        }

    def _pad(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return x
        if self.padding_mode == "constant":
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)
        if self.padding_mode == "reflect":
            height, width = x.shape[-2:]
            invalid = (pad_h > 0 and pad_h >= height) or (pad_w > 0 and pad_w >= width)
            if invalid:
                if self.strict_padding:
                    raise ValueError(
                        "HOSQ reflect padding requires the spatial dimension to exceed "
                        f"the radius. Got feature map {height}x{width} and padding "
                        f"({pad_h},{pad_w}). Use PADDING='replicate' explicitly for "
                        "small patch grids such as ViT-B/32."
                    )
                return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=self.padding_mode)

    def _conv_axis(
        self,
        x: torch.Tensor,
        axis_name: str,
        kernel_1d: torch.Tensor,
    ) -> torch.Tensor:
        kernel_size = int(kernel_1d.shape[-1])
        radius = kernel_size // 2
        if axis_name == "h":
            weight = kernel_1d.view(self.C, 1, kernel_size, 1)
            padded = self._pad(x, pad_h=radius, pad_w=0)
        elif axis_name == "w":
            weight = kernel_1d.view(self.C, 1, 1, kernel_size)
            padded = self._pad(x, pad_h=0, pad_w=radius)
        else:
            raise ValueError(f"unknown axis {axis_name!r}")
        return F.conv2d(padded, weight, stride=1, padding=0, groups=self.C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"HOSQDT1DAdapter expects BCHW input, got {tuple(x.shape)}")
        if x.shape[1] != self.C:
            raise ValueError(f"channel mismatch: expected {self.C}, got {x.shape[1]}")

        kernels = self.build_normalized_kernels(x.device, x.dtype)
        response = torch.zeros_like(x)
        for axis_index, axis_name in enumerate(self.axis_names):
            response = response + self._conv_axis(x, axis_name, kernels[axis_index])
        response = self.pw(response)
        return x + self.residual_scale * self.gate * response


class HOSQTokenAdapter(nn.Module):
    """Apply HOSQ-DT1D to ViT patch tokens while preserving prefix tokens."""

    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int],
        num_prefix_tokens: int = 1,
        **hosq_kwargs,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.num_prefix_tokens = int(num_prefix_tokens)
        if self.grid_size[0] <= 0 or self.grid_size[1] <= 0:
            raise ValueError(f"invalid grid_size {self.grid_size}")
        if self.num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must be non-negative")
        self.hosq = HOSQDT1DAdapter(C=self.embed_dim, **hosq_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"HOSQTokenAdapter expects BND input, got {tuple(x.shape)}")
        batch, token_count, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(f"embed_dim mismatch: expected {self.embed_dim}, got {channels}")
        expected = self.num_prefix_tokens + self.grid_size[0] * self.grid_size[1]
        if token_count != expected:
            raise ValueError(
                f"token count {token_count} does not match {self.num_prefix_tokens} prefix "
                f"tokens plus grid {self.grid_size[0]}x{self.grid_size[1]} ({expected})"
            )
        prefix = x[:, :self.num_prefix_tokens, :]
        patches = x[:, self.num_prefix_tokens:, :]
        feature_map = patches.transpose(1, 2).reshape(
            batch, channels, self.grid_size[0], self.grid_size[1]
        )
        adapted = self.hosq(feature_map)
        adapted_patches = adapted.reshape(
            batch, channels, self.grid_size[0] * self.grid_size[1]
        ).transpose(1, 2)
        return torch.cat([prefix, adapted_patches], dim=1)

    def parameter_count_breakdown(self) -> Dict[str, int]:
        return self.hosq.parameter_count_breakdown()

    def mathematical_invariants(self) -> Dict[str, float]:
        return self.hosq.mathematical_invariants()


# Backward-compatible aliases used by old code/configs.
HCCAdapter = DT1DAdapter
H1D_DT_Adapter = DT1DAdapter
OneDDTAdapter = DT1DAdapter
DT1DTokenAdapter = HCCTokenAdapter
HOSQAdapter = HOSQDT1DAdapter
