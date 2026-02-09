# src/models/vit_adapter/hcc_adapter.py
from typing import Tuple, Optional
from math import gcd

import torch
import torch.nn as nn
import torch.nn.functional as F


class HCCAdapter(nn.Module):
    """
    Hartley–Cosine (even) shift aggregation via depthwise dilated conv on feature maps.

    Updated to match DT1DAdapter:
      - group-shared alpha via alpha_group (channels-per-alpha-group)
      - optional grouped PW bottleneck mixing (pw_groups)
      - new API: no_pw (inverse of legacy use_pw)
      - legacy compatibility: per_channel -> alpha_group=1, use_pw -> no_pw
      - padding fixed (no value=None for reflect/replicate)
    """
    def __init__(
        self,
        C: int,
        M: int = 1,
        h: int = 1,
        axis: str = "hw",

        # ---- NEW (DT1D-style) ----
        alpha_group: int = 16,          # channels per alpha group
        tie_sym: bool = True,
        no_pw: bool = False,
        pw_ratio: int = 32,
        pw_groups: int = 4,
        use_bn: bool = False,
        residual_scale: float = 1.0,
        gate_init: float = 0.1,
        padding_mode: str = "reflect",

        # ---- legacy knobs (optional) ----
        per_channel: Optional[bool] = None,  # if True -> alpha_group=1
        use_pw: Optional[bool] = None,       # if provided -> no_pw = not use_pw

        **legacy,  # swallow any unknown legacy kwargs
    ):
        super().__init__()
        assert axis in ("h", "w", "hw")
        self.C, self.M, self.h = int(C), int(M), int(h)
        self.axis = axis
        self.tie_sym = bool(tie_sym)
        self.padding_mode = str(padding_mode)
        self.residual_scale = float(residual_scale)

        # ---- translate legacy args ----
        # legacy: per_channel=True -> alpha_group=1
        if per_channel is not None:
            alpha_group = 1 if bool(per_channel) else int(alpha_group)
        # legacy: use_pw=True -> no_pw=False
        if use_pw is not None:
            no_pw = (not bool(use_pw))
        # ignore any other legacy keys silently

        self.alpha_group = max(1, int(alpha_group))
        self.no_pw = bool(no_pw)
        self.use_bn = bool(use_bn)

        # ---------- α coefficients (group-shared) ----------
        # Interpret alpha_group as "channels per group"
        # Number of groups: G = floor(C / alpha_group), at least 1
        G = max(1, self.C // self.alpha_group)
        ncoef = self.M + 1  # center + M side taps
        self.alpha = nn.Parameter(torch.zeros(G, ncoef))
        with torch.no_grad():
            self.alpha[:, 0].fill_(1.0)  # identity-safe init

        # ---------- optional channel mixing via PW (grouped) ----------
        if not self.no_pw:
            Hhid = max(1, self.C // max(1, int(pw_ratio)))

            g = max(1, int(pw_groups))
            g = min(g, self.C, Hhid)

            # ensure groups divide both C and Hhid
            g = gcd(g, self.C)
            g = gcd(g, Hhid) or 1
            self.pw_groups = g

            self.pw = nn.Sequential(
                nn.Conv2d(self.C, Hhid, 1, groups=g, bias=False),
                nn.BatchNorm2d(Hhid) if self.use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(Hhid, self.C, 1, groups=g, bias=False),
                nn.BatchNorm2d(self.C) if self.use_bn else nn.Identity(),
            )
        else:
            self.pw = nn.Identity()

        # ---------- global residual gate ----------
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def _build_even_kernel_1d(self, device, dtype) -> torch.Tensor:
        """
        Build symmetric 1D kernel of length K = 2M+1 from group-shared alpha.
        Returns depthwise conv weight of shape (C, 1, K).
        """
        K = 2 * self.M + 1
        center = self.M

        G = max(1, self.C // self.alpha_group)
        wg = torch.zeros(G, K, device=device, dtype=dtype)

        # center + symmetric side taps
        wg[:, center] = self.alpha[:, 0]
        for m in range(1, self.M + 1):
            val = self.alpha[:, m]
            wg[:, center - m] = val
            # NOTE: tie_sym kept for API compatibility; kernel is symmetric unless you
            # introduce separate right-tap params.
            wg[:, center + m] = val if self.tie_sym else val

        # L1-normalize per group (stable)
        s = wg.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        wg = wg / s

        # expand group kernels to channels
        reps = [self.alpha_group] * G
        reps[-1] = self.C - self.alpha_group * (G - 1)  # handle remainder channels
        w = torch.cat(
            [wg[i].unsqueeze(0).repeat(reps[i], 1) for i in range(G)],
            dim=0
        )  # (C, K)

        return w.unsqueeze(1)  # (C, 1, K)

    def _pad(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        pads = (pad_w, pad_w, pad_h, pad_h)
        mode = self.padding_mode
        if mode == "reflect":
            return F.pad(x, pads, mode="reflect")
        if mode == "replicate":
            return F.pad(x, pads, mode="replicate")
        # constant/zero padding fallback
        return F.pad(x, pads, mode="constant", value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.C, f"channel mismatch: got {C}, expected {self.C}"

        w1d = self._build_even_kernel_1d(x.device, x.dtype)  # (C,1,K)
        K = 2 * self.M + 1

        y = torch.zeros_like(x)

        if "h" in self.axis:
            wh = w1d.view(self.C, 1, K, 1)
            xh = self._pad(x, pad_h=self.M * self.h, pad_w=0)
            yh = F.conv2d(
                xh, wh,
                stride=1, padding=0,
                dilation=(self.h, 1),
                groups=self.C
            )
            y = y + yh

        if "w" in self.axis:
            ww = w1d.view(self.C, 1, 1, K)
            xw = self._pad(x, pad_h=0, pad_w=self.M * self.h)
            yw = F.conv2d(
                xw, ww,
                stride=1, padding=0,
                dilation=(1, self.h),
                groups=self.C
            )
            y = y + yw

        y = self.pw(y)
        return x + self.residual_scale * self.gate * y


class HCCTokenAdapter(nn.Module):
    """
    Wraps HCCAdapter to operate on ViT tokens:
      x: (B, N, D), N = 1 + H*W (class token first)
      grid_size: (H, W)

    Only patch tokens are transformed; class token is passthrough.

    Updated to accept DT1D-style args (alpha_group/no_pw/pw_groups) and legacy
    args (per_channel/use_pw) for compatibility.
    """
    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int],
        M: int = 1,
        h: int = 1,
        axis: str = "hw",

        # DT1D-style
        alpha_group: int = 16,
        tie_sym: bool = True,
        no_pw: bool = True,          # default matches your old HCCTokenAdapter(use_pw=False)
        pw_ratio: int = 32,
        pw_groups: int = 4,
        use_bn: bool = False,
        residual_scale: float = 1.0,
        gate_init: float = 0.1,
        padding_mode: str = "reflect",

        # legacy compatibility
        per_channel: Optional[bool] = None,
        use_pw: Optional[bool] = None,

        **legacy,
    ):
        super().__init__()
        self.D = int(embed_dim)
        gh, gw = int(grid_size[0]), int(grid_size[1])
        assert gh > 0 and gw > 0, f"invalid grid_size: {grid_size}"
        self.gh, self.gw = gh, gw

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
        """
        x: (B, N, D) with cls at position 0, patches follow
        """
        B, N, D = x.shape
        assert D == self.D, f"embed_dim mismatch: got {D}, expected {self.D}"
        expected = 1 + self.gh * self.gw
        assert N == expected, f"N={N} but grid {self.gh}x{self.gw} implies {expected}"

        cls_tok = x[:, :1, :]  # (B,1,D)
        patches = x[:, 1:, :]  # (B,H*W,D)

        fmap = patches.transpose(1, 2).reshape(B, D, self.gh, self.gw)  # (B,D,H,W)
        fmap_out = self.hcc(fmap)  # (B,D,H,W)
        patches_out = fmap_out.reshape(B, D, self.gh * self.gw).transpose(1, 2)  # (B,H*W,D)

        return torch.cat([cls_tok, patches_out], dim=1)
