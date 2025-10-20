"""Device selection utilities for Torch models."""
from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover - torch installed in notebook runtime
    torch = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceConfig:
    """Container describing the chosen accelerator."""

    device: str
    torch_dtype: str
    is_cuda: bool
    device_index: int | None


def detect_torch_device(prefer_cuda: bool = True) -> DeviceConfig:
    """Return the recommended Torch device configuration.

    Parameters
    ----------
    prefer_cuda:
        When ``True`` (default) a CUDA capable accelerator is preferred when
        available. The function gracefully falls back to CPU when torch is not
        installed or a GPU cannot be accessed.
    """

    if torch is None:
        LOGGER.warning("PyTorch is not available; defaulting to CPU execution")
        return DeviceConfig("cpu", "float32", False, None)

    if prefer_cuda and torch.cuda.is_available():
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        LOGGER.info("Using CUDA device %s (%s)", index, name)
        return DeviceConfig(f"cuda:{index}", "float32", True, index)

    if prefer_cuda and torch.backends.mps.is_available():  # pragma: no cover - macOS specific
        LOGGER.info("Using Apple Metal Performance Shaders (MPS) backend")
        return DeviceConfig("mps", "float32", True, None)

    LOGGER.info("Using CPU device for Torch execution")
    return DeviceConfig("cpu", "float32", False, None)


__all__ = ["DeviceConfig", "detect_torch_device"]
