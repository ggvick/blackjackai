"""Utility helpers for GPU-first Rainbow DQN training."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Dict

import numpy as np
import torch


def assert_cuda_available() -> torch.device:
    """Return the CUDA device if available, otherwise raise a helpful error."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. In Colab: Runtime → Change runtime type → GPU."
        )
    return torch.device("cuda")


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU & CUDA) RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_compile(module: torch.nn.Module, enabled: bool = True) -> torch.nn.Module:
    """Compile the module when requested and supported by the runtime."""

    if not enabled or not hasattr(torch, "compile"):
        return module
    return torch.compile(module, mode="default")  # type: ignore[attr-defined]


def safe_state_dict_from_module(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a state_dict that works for compiled and non-compiled modules."""

    base = getattr(module, "_orig_mod", module)
    return base.state_dict()


def strip_orig_mod_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the `_orig_mod.` prefix added by torch.compile when present."""

    return {
        (key[10:] if key.startswith("_orig_mod.") else key): value
        for key, value in state_dict.items()
    }


@contextmanager
def autocast_if(enabled: bool):
    """Context manager that enables CUDA autocast when requested."""

    if enabled:
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


__all__ = [
    "assert_cuda_available",
    "seed_everything",
    "maybe_compile",
    "safe_state_dict_from_module",
    "strip_orig_mod_prefix",
    "autocast_if",
]
