"""Minimal proposal-method checks for the ViT DT1D/HCC token adapter."""
import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "hcc_adapter_direct",
    _ROOT / "src" / "models" / "vit_adapter" / "hcc_adapter.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
HCCAdapter = _MOD.HCCAdapter
HCCTokenAdapter = _MOD.HCCTokenAdapter


def test_weighted_hcc_kernel_is_2m_plus_3_and_l1_normalized():
    m = HCCAdapter(C=16, M=2, dilations="1,2,4", scale_adaptive=True, axis="hw", alpha_group=8)
    k = m._build_weighted_hcc_kernel_1d(0, 0, torch.device("cpu"), torch.float32)
    assert k.shape == (16, 1, 7)
    assert torch.all(k.squeeze(1).abs().sum(dim=1) <= 1.00001)


def test_static_axis_scale_gate_shape_and_no_router():
    m = HCCAdapter(C=16, M=1, dilations="1,2,4", scale_adaptive=True, axis="hw", input_adaptive_gate=True)
    assert m.axis_scale_router is None
    assert m.input_adaptive_gate is False
    w = m.axis_scale_weights()
    assert w.shape == (2, 3)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)


def test_spatial_forward_is_identity_at_zero_gate():
    m = HCCAdapter(C=8, M=1, dilations="1,2", scale_adaptive=True, axis="hw", gate_init=0.0, no_pw=True)
    x = torch.randn(2, 8, 8, 8)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x)


def test_token_adapter_keeps_cls_token_and_backprops():
    torch.manual_seed(0)
    m = HCCTokenAdapter(embed_dim=8, grid_size=(4, 4), M=1, dilations="1,2", scale_adaptive=True, axis="hw", gate_init=0.01)
    x = torch.randn(2, 17, 8, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y[:, 0], x[:, 0])
    y.mean().backward()
    assert x.grad is not None
