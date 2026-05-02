import torch

from src.models.vit_adapter.hcc_adapter import HCCAdapter, HCCTokenAdapter


def test_hcc_identity_when_gate_zero():
    adapter = HCCAdapter(C=7, M=2, h=1, axis="hw", alpha_group=3, gate_init=0.0)
    x = torch.randn(2, 7, 4, 5)
    y = adapter(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_hcc_ceil_grouping_handles_remainder_channels():
    adapter = HCCAdapter(C=10, M=1, alpha_group=4)
    assert adapter.num_alpha_groups == 3
    w = adapter._build_even_kernel_1d(torch.device("cpu"), torch.float32)
    assert w.shape == (10, 1, 3)


def test_hcc_hw_average_not_double_sum():
    # With identity kernel and gate=1, axis='hw' should produce x + average(x, x) = 2x.
    # If H/W were summed without averaging, it would produce 3x.
    adapter = HCCAdapter(C=3, M=0, axis="hw", alpha_group=2, gate_init=1.0, no_pw=True)
    x = torch.randn(1, 3, 2, 2)
    y = adapter(x)
    assert torch.allclose(y, 2.0 * x, atol=1e-6)


def test_token_adapter_preserves_class_token():
    adapter = HCCTokenAdapter(embed_dim=6, grid_size=(2, 3), M=1, alpha_group=4, gate_init=1.0)
    x = torch.randn(2, 7, 6)
    y = adapter(x)
    assert y.shape == x.shape
    assert torch.allclose(y[:, 0], x[:, 0], atol=1e-6)
