"""Shared helpers for Blackjack RL notebook."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - satisfied in runtime
    torch = None  # type: ignore


@dataclass
class TimingInfo:
    label: str
    duration_seconds: float


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - requires GPU
            torch.cuda.manual_seed_all(seed)


def running_mean(values: Iterable[float], window: int) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=np.float64)
    if window <= 1:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    result = (cumsum[window:] - cumsum[:-window]) / float(window)
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, result])


def bankroll_targets_reached(bankroll: float, lower: float, upper: float) -> bool:
    return bankroll <= lower or bankroll >= upper


def to_json(data: dict, path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x -= np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def log_uniform(low: float, high: float, size: int) -> np.ndarray:
    log_low = math.log(low)
    log_high = math.log(high)
    samples = np.random.uniform(log_low, log_high, size=size)
    return np.exp(samples)


__all__ = [
    "TimingInfo",
    "ensure_dir",
    "set_global_seed",
    "running_mean",
    "bankroll_targets_reached",
    "to_json",
    "softmax",
    "log_uniform",
]
