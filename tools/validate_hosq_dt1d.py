#!/usr/bin/env python3
"""Standalone HOSQ mathematical and gradient smoke validation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


def load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "models" / "vit_adapter" / "hcc_adapter.py"
    spec = importlib.util.spec_from_file_location("hcc_adapter_validate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    module = load_module()
    hosq = module.HOSQTokenAdapter(
        embed_dim=768,
        grid_size=(14, 14),
        axis="hw",
        coarse_group=32,
        subgroup_size=8,
        rank4=1,
        rank8=2,
        no_pw=True,
        gate_init=0.01,
        padding_mode="reflect",
    )
    x = torch.randn(1, 197, 768, requires_grad=True)
    y = hosq(x)
    y.mean().backward()
    report = {
        "output_shape": list(y.shape),
        "class_token_exact": bool(torch.equal(y[:, 0], x[:, 0])),
        "parameter_breakdown": hosq.parameter_count_breakdown(),
        "mathematical_invariants": hosq.mathematical_invariants(),
        "gradients": {
            "input": x.grad is not None,
            "coarse": hosq.hosq.quotient_beta.grad is not None,
            "detail4": hosq.hosq.detail4.grad is not None,
            "detail8": hosq.hosq.detail8.grad is not None,
            "gate": hosq.hosq.gate.grad is not None,
        },
    }
    if report["parameter_breakdown"]["total"] != 385:
        raise RuntimeError(report)
    if report["mathematical_invariants"]["maximum_joint_axis_l1"] > 1.000001:
        raise RuntimeError(report)
    if not all(report["gradients"].values()) or not report["class_token_exact"]:
        raise RuntimeError(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
