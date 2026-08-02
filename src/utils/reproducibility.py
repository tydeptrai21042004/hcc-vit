#!/usr/bin/env python3
"""Reproducibility helpers for multi-seed experiments."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_reproducible_seed(seed: Optional[int], deterministic: bool = True) -> int:
    """Seed Python, NumPy, CPU/CUDA PyTorch, and deterministic backends.

    ``None`` is converted to zero so DataLoader construction always has an
    explicit seed.  The returned integer is the seed that was applied.
    """
    resolved = 0 if seed is None else int(seed)
    os.environ["PYTHONHASHSEED"] = str(resolved)
    random.seed(resolved)
    np.random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(resolved)
        torch.cuda.manual_seed_all(resolved)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older PyTorch
            torch.use_deterministic_algorithms(True)
    return resolved


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python inside a DataLoader worker from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: Optional[int]) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(0 if seed is None else int(seed))
    return generator
