#!/usr/bin/env python3
"""
Reviewer-facing efficiency profiler for transformer PEFT experiments.

Reports:
  * total/trainable/frozen parameters
  * trainable percentage
  * FLOPs / GFLOPs when fvcore is installed
  * inference latency and throughput
  * peak inference memory
  * optional dummy training-step time and peak training memory

Example:
  python tools/profile_efficiency.py \
    --config-file configs/peft_transformer/flowers_hcc_dt1d.yaml \
    --no-pretrain \
    DATA.NUMBER_CLASSES 102
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.configs.config import get_cfg  # noqa: E402
from src.models.vit_models import ViT  # noqa: E402


def count_params(model: torch.nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "trainable_percent": float(trainable / max(1, total) * 100.0),
    }


def try_count_flops(model: torch.nn.Module, dummy: torch.Tensor) -> Dict[str, Optional[float]]:
    try:
        from fvcore.nn import FlopCountAnalysis

        model.eval()
        flops = FlopCountAnalysis(model, dummy).total()
        return {"flops": float(flops), "gflops": float(flops / 1e9), "flops_available": True}
    except Exception as exc:
        return {
            "flops": None,
            "gflops": None,
            "flops_available": False,
            "flops_error": str(exc),
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference(model: torch.nn.Module, dummy: torch.Tensor, warmup: int, repeat: int) -> Dict[str, float]:
    model.eval()
    device = dummy.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(dummy)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(max(1, repeat)):
            _ = model(dummy)
        _sync(device)
        elapsed = time.perf_counter() - t0

    latency = elapsed / max(1, repeat)
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "inference_latency_ms": float(latency * 1000.0),
        "inference_throughput_img_s": float(dummy.shape[0] / max(latency, 1e-12)),
        "peak_inference_memory_mb": float(peak_mem_mb),
    }


def benchmark_train_step(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    num_classes: int,
    steps: int,
) -> Dict[str, float]:
    device = dummy.device
    model.train()
    labels = torch.randint(0, int(num_classes), (dummy.shape[0],), device=device)
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(max(1, steps)):
        optim.zero_grad(set_to_none=True)
        loss = criterion(model(dummy), labels)
        loss.backward()
        optim.step()
    _sync(device)
    elapsed = time.perf_counter() - t0
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "dummy_train_step_ms": float(elapsed / max(1, steps) * 1000.0),
        "peak_train_memory_mb": float(peak_mem_mb),
    }


def build_cfg(args: argparse.Namespace):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()
    return cfg


def write_outputs(metrics: Dict[str, Any], output_dir: str, json_name: str, csv_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, json_name)
    csv_path = os.path.join(output_dir, csv_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-pretrain", action="store_true", help="Build architecture without loading pretrained checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--json-name", default="efficiency_profile.json")
    parser.add_argument("--csv-name", default="efficiency_profile.csv")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = build_cfg(args)
    device = torch.device(args.device)
    model = ViT(cfg, load_pretrain=not args.no_pretrain).to(device)

    batch_size = int(args.batch_size or cfg.PROFILE.BATCH_SIZE)
    warmup = int(args.warmup if args.warmup is not None else cfg.PROFILE.WARMUP)
    repeat = int(args.repeat if args.repeat is not None else cfg.PROFILE.REPEAT)
    train_steps = int(args.train_steps if args.train_steps is not None else cfg.PROFILE.TRAIN_STEPS)
    dummy = torch.randn(batch_size, 3, int(cfg.DATA.CROPSIZE), int(cfg.DATA.CROPSIZE), device=device)

    metrics: Dict[str, Any] = {
        "dataset": cfg.DATA.NAME,
        "feature": cfg.DATA.FEATURE,
        "transfer_type": cfg.MODEL.TRANSFER_TYPE,
        "adapter_name": str(getattr(cfg.MODEL.ADAPTER, "NAME", "none")),
        "crop_size": int(cfg.DATA.CROPSIZE),
        "batch_size": int(batch_size),
        "device": str(device),
    }
    metrics.update(count_params(model))
    metrics.update(try_count_flops(model, dummy))
    metrics.update(benchmark_inference(model, dummy, warmup=warmup, repeat=repeat))
    if train_steps > 0:
        metrics.update(benchmark_train_step(model, dummy, cfg.DATA.NUMBER_CLASSES, steps=train_steps))

    output_dir = args.output_dir or cfg.OUTPUT_DIR
    write_outputs(metrics, output_dir, args.json_name, args.csv_name)


if __name__ == "__main__":
    main()
