# src/models/vit_adapter/hcc_adapter.py
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HCCAdapter(nn.Module):
    """
    Hartley–Cosine (even) shift aggregation via depthwise dilated conv on feature maps.
      - Axis: 'h', 'w', or 'hw' (sum across axes).
      - Tie ±m weights; learn per-channel alphas (center + M side taps).
      - Optional pointwise bottleneck mixing + BN.
      - Reflect/replicate/zero padding.
    """
    def __init__(
        self, C: int, M: int = 1, h: int = 1, axis: str = "hw",
        per_channel: bool = True, tie_sym: bool = True,
        use_pw: bool = True, pw_ratio: int = 8, use_bn: bool = True,
        residual_scale: float = 1.0, gate_init: float = 0.1,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        assert axis in ("h", "w", "hw")
        self.C = int(C)
        self.M = int(M)
        self.h = int(h)
        self.axis = axis
        self.tie_sym = tie_sym
        self.per_channel = per_channel
        self.padding_mode = padding_mode
        self.residual_scale = float(residual_scale)

        # alpha: (C, M+1) or (M+1,)
        ncoef = self.M + 1
        if per_channel:
            self.alpha = nn.Parameter(torch.zeros(self.C, ncoef))
        else:
            self.alpha = nn.Parameter(torch.zeros(ncoef))

        # Identity-safe init (center ≈ 1, sides ≈ 0)
        with torch.no_grad():
            if per_channel:
                self.alpha[:, 0].fill_(1.0)
            else:
                self.alpha[0] = 1.0

        # Optional channel mixing (DW -> PW bottleneck -> PW expand)
        self.use_pw = use_pw
        if use_pw:
            hid = max(1, self.C // pw_ratio)
            self.pw = nn.Sequential(
                nn.Conv2d(self.C, hid, 1, bias=False),
                nn.BatchNorm2d(hid) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid, self.C, 1, bias=False),
                nn.BatchNorm2d(self.C) if use_bn else nn.Identity(),
            )
        else:
            self.pw = nn.Identity()

        # Global gate
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def _build_even_kernel_1d(self, device, dtype):
        """
        Build symmetric 1D kernel of length K = 2M+1 from alpha (center + M sides).
        Returns depthwise conv weight of shape (C, 1, K).
        """
        K = 2 * self.M + 1
        center = self.M

        if self.per_channel:
            w = torch.zeros(self.C, K, device=device, dtype=dtype)
            w[:, center] = self.alpha[:, 0]
            for m in range(1, self.M + 1):
                # tie ±m (default)
                val = self.alpha[:, m]
                w[:, center - m] = val
                w[:, center + m] = val
            # L1-normalize per channel
            s = w.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
            w = w / s
            return w.unsqueeze(1)  # (C,1,K)
        else:
            w = torch.zeros(K, device=device, dtype=dtype)
            w[center] = self.alpha[0]
            for m in range(1, self.M + 1):
                val = self.alpha[m]
                w[center - m] = val
                w[center + m] = val
            s = w.abs().sum().clamp_min(1e-6)
            w = (w / s).view(1, 1, K).repeat(self.C, 1, 1)  # (C,1,K)
            return w

    def _pad(self, x, pad_h: int, pad_w: int):
        if self.padding_mode == "reflect":
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        if self.padding_mode == "replicate":
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.C, f"channel mismatch: got {C}, expected {self.C}"
        w1d = self._build_even_kernel_1d(x.device, x.dtype)  # (C,1,K)

        y = torch.zeros_like(x)  # safer than starting from literal 0
        K = 2 * self.M + 1

        if "h" in self.axis:
            wh = w1d.view(self.C, 1, K, 1)
            xh = self._pad(x, pad_h=self.M * self.h, pad_w=0)
            yh = F.conv2d(xh, wh, stride=1, padding=0, dilation=(self.h, 1), groups=self.C)
            y = y + yh

        if "w" in self.axis:
            ww = w1d.view(self.C, 1, 1, K)
            xw = self._pad(x, pad_h=0, pad_w=self.M * self.h)
            yw = F.conv2d(xw, ww, stride=1, padding=0, dilation=(1, self.h), groups=self.C)
            y = y + yw

        y = self.pw(y)
        return x + self.residual_scale * self.gate * y


class HCCTokenAdapter(nn.Module):
    """
    Wraps HCCAdapter to operate on ViT tokens:
      x: (B, N, D), N = 1 + H*W (class token first)
      grid_size: (H, W)
    Only patch tokens are transformed; class token is passthrough.
    """
    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int],
        M: int = 1,
        h: int = 1,
        axis: str = "hw",
        per_channel: bool = True,
        tie_sym: bool = True,
        use_pw: bool = False,
        pw_ratio: int = 8,
        use_bn: bool = True,
        residual_scale: float = 1.0,
        gate_init: float = 0.1,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        self.D = int(embed_dim)
        gh, gw = int(grid_size[0]), int(grid_size[1])
        assert gh > 0 and gw > 0, f"invalid grid_size: {grid_size}"
        self.gh, self.gw = gh, gw

        self.hcc = HCCAdapter(
            C=self.D, M=M, h=h, axis=axis, per_channel=per_channel,
            tie_sym=tie_sym, use_pw=use_pw, pw_ratio=pw_ratio, use_bn=use_bn,
            residual_scale=residual_scale, gate_init=gate_init, padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D) with cls at position 0, patches follow
        """
        B, N, D = x.shape
        assert D == self.D, f"embed_dim mismatch: got {D}, expected {self.D}"
        expected = 1 + self.gh * self.gw
        assert N == expected, f"N={N} but grid {self.gh}x{self.gw} implies {expected}"

        cls_tok = x[:, :1, :]               # (B,1,D)
        patches = x[:, 1:, :]               # (B,H*W,D)
        fmap = patches.transpose(1, 2).reshape(B, D, self.gh, self.gw)  # -> (B,D,H,W)

        fmap_out = self.hcc(fmap)           # (B,D,H,W)

        patches_out = fmap_out.reshape(B, D, self.gh * self.gw).transpose(1, 2)  # (B,H*W,D)
        return torch.cat([cls_tok, patches_out], dim=1)
