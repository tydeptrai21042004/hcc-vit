from types import SimpleNamespace

import torch
import torch.nn as nn

from src.models.vit_adapter.lora import LoRALinear, apply_lora_to_attention
from src.models.vit_adapter.peft_modules import AdaptFormerAdapter, SSF


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(8, 8)
        self.key = nn.Linear(8, 8)
        self.value = nn.Linear(8, 8)
        self.out = nn.Linear(8, 8)


def test_lora_replaces_selected_attention_layers_and_freezes_base():
    attn = TinyAttention()
    apply_lora_to_attention(attn, rank=2, alpha=4, targets="query,value")
    assert isinstance(attn.query, LoRALinear)
    assert isinstance(attn.value, LoRALinear)
    assert not isinstance(attn.key, LoRALinear)
    assert all(not p.requires_grad for p in attn.query.base.parameters())
    x = torch.randn(3, 5, 8)
    assert attn.query(x).shape == (3, 5, 8)


def test_adaptformer_adapter_shape():
    module = AdaptFormerAdapter(hidden_size=8, reduction_factor=4)
    x = torch.randn(2, 5, 8)
    y = module(x)
    assert y.shape == x.shape


def test_ssf_shape_and_identity_initialization():
    module = SSF(hidden_size=8, init_scale=1.0, init_shift=0.0)
    x = torch.randn(2, 5, 8)
    y = module(x)
    assert torch.allclose(y, x, atol=1e-6)


def _tiny_config():
    return SimpleNamespace(
        hidden_size=8,
        transformer={
            "num_heads": 2,
            "attention_dropout_rate": 0.0,
            "dropout_rate": 0.0,
            "mlp_dim": 16,
        },
    )


def test_module_parameter_names_are_visible_to_adapter_freezer():
    lora = LoRALinear(nn.Linear(8, 8), rank=2, alpha=4.0)
    adapt = AdaptFormerAdapter(hidden_size=8, reduction_factor=4)
    ssf = SSF(hidden_size=8)
    names = []
    names += ["lora." + n for n, _ in lora.named_parameters()]
    names += ["adaptformer_adapter." + n for n, _ in adapt.named_parameters()]
    names += ["ssf." + n for n, _ in ssf.named_parameters()]
    assert any("lora_" in n for n in names)
    assert any("adapter" in n for n in names)
    assert any("ssf_" in n for n in names)
