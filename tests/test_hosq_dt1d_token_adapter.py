"""Deterministic mathematical and execution tests for HOSQ-DT1D."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "hcc_adapter_hosq_direct",
    _ROOT / "src" / "models" / "vit_adapter" / "hcc_adapter.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
HOSQDT1DAdapter = _MOD.HOSQDT1DAdapter
HOSQTokenAdapter = _MOD.HOSQTokenAdapter


def test_hosq_basis_is_zero_mean_and_orthonormal():
    basis = HOSQDT1DAdapter._orthogonal_subgroup_basis(4)
    assert basis.shape == (4, 3)
    assert torch.allclose(basis.sum(dim=0), torch.zeros(3), atol=1e-7)
    assert torch.allclose(basis.T @ basis, torch.eye(3), atol=1e-7)


def test_hosq_vitb_parameter_budget_is_385_per_block():
    module = HOSQDT1DAdapter(
        C=768,
        axis="hw",
        coarse_group=32,
        subgroup_size=8,
        rank4=1,
        rank8=2,
        no_pw=True,
    )
    breakdown = module.parameter_count_breakdown()
    assert breakdown["coarse_quotient"] == 240
    assert breakdown["detail_offset4"] == 48
    assert breakdown["detail_offset8"] == 96
    assert breakdown["residual_gate"] == 1
    assert breakdown["total"] == 385
    assert 12 * breakdown["total"] == 4620


def test_zero_details_reduce_to_group32_coarse_kernel():
    torch.manual_seed(3)
    module = HOSQDT1DAdapter(C=64, coarse_group=32, subgroup_size=8)
    with torch.no_grad():
        module.quotient_beta.normal_()
        module.detail4.zero_()
        module.detail8.zero_()
    kernels = module.build_normalized_kernels(torch.device("cpu"), torch.float32)
    # All 32 channels in each coarse group have the same kernel when details are zero.
    assert torch.allclose(kernels[:, 0], kernels[:, 31], atol=1e-7)
    assert torch.allclose(kernels[:, 32], kernels[:, 63], atol=1e-7)


def test_detail_atoms_are_zero_dc_and_jointly_bounded():
    module = HOSQDT1DAdapter(C=32, coarse_group=32, subgroup_size=8)
    with torch.no_grad():
        module.quotient_beta.zero_()
        module.detail4.fill_(0.7)
        module.detail8.fill_(-0.4)
    kernels = module.build_normalized_kernels(torch.device("cpu"), torch.float32).squeeze(2)
    assert torch.allclose(kernels.sum(dim=-1), torch.zeros_like(kernels[..., 0]), atol=1e-6)
    joint_l1 = kernels.abs().sum(dim=-1).sum(dim=0)
    assert torch.all(joint_l1 <= 1.000001)


def test_hosq_token_adapter_preserves_cls_and_backpropagates():
    torch.manual_seed(4)
    module = HOSQTokenAdapter(
        embed_dim=32,
        grid_size=(14, 14),
        coarse_group=32,
        subgroup_size=8,
        gate_init=0.01,
    )
    x = torch.randn(2, 197, 32, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.equal(y[:, 0], x[:, 0])
    y.square().mean().backward()
    assert x.grad is not None
    assert module.hosq.quotient_beta.grad is not None
    assert module.hosq.detail4.grad is not None
    assert module.hosq.detail8.grad is not None
    assert module.hosq.gate.grad is not None


def test_hosq_executes_one_depthwise_convolution_per_axis():
    module = HOSQDT1DAdapter(C=8, axis="hw", coarse_group=8, subgroup_size=2)
    x = torch.randn(1, 8, 17, 17)
    original = _MOD.F.conv2d
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with patch.object(_MOD.F, "conv2d", side_effect=counted):
        module(x)
    assert len(calls) == 2


def test_reflect_padding_rejects_patch32_grid_in_strict_mode():
    module = HOSQTokenAdapter(
        embed_dim=8,
        grid_size=(7, 7),
        coarse_group=8,
        subgroup_size=2,
        padding_mode="reflect",
        strict_padding=True,
    )
    with pytest.raises(ValueError, match="reflect padding"):
        module(torch.randn(1, 50, 8))


def test_replicate_padding_supports_patch32_grid():
    module = HOSQTokenAdapter(
        embed_dim=8,
        grid_size=(7, 7),
        coarse_group=8,
        subgroup_size=2,
        padding_mode="replicate",
    )
    x = torch.randn(1, 50, 8)
    assert module(x).shape == x.shape
