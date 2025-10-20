"""Utilities for handling legal action masks."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def legal_action_mask(legal_actions: Sequence[int], num_actions: int = 5) -> np.ndarray:
    """Return a binary mask for legal actions.

    Parameters
    ----------
    legal_actions:
        Iterable of action indices that are valid in the current state.
    num_actions:
        Total number of discrete actions.  Defaults to five play actions
        (stand, hit, double, split, surrender).
    """

    mask = np.zeros(num_actions, dtype=np.float32)
    for action in legal_actions:
        if 0 <= int(action) < num_actions:
            mask[int(action)] = 1.0
    return mask


def apply_action_mask(
    q_values: np.ndarray, mask: np.ndarray, invalid_fill: float = -1e9
) -> np.ndarray:
    """Apply an action mask to a Q-value vector.

    Invalid actions are filled with ``invalid_fill`` so an ``argmax`` will never
    select them.  A copy of the Q-values is returned, leaving the input array
    untouched.
    """

    if q_values.shape != mask.shape:
        raise ValueError("Shape mismatch between q_values and mask")
    adjusted = np.array(q_values, copy=True)
    adjusted[mask < 0.5] = invalid_fill
    return adjusted


__all__ = ["legal_action_mask", "apply_action_mask"]
