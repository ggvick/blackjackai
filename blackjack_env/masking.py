"""Action masking utilities for Blackjack."""

from __future__ import annotations

import numpy as np

from .utils import Hand

ACTIONS = ["hit", "stand", "double", "split", "surrender"]


class Action:
    HIT = 0
    STAND = 1
    DOUBLE = 2
    SPLIT = 3
    SURRENDER = 4


def legal_action_mask(
    player_hand: Hand,
    stage: str,
    allow_double: bool,
    allow_split: bool,
    allow_surrender: bool,
    max_splits: int,
    splits_used: int,
) -> np.ndarray:
    mask = np.zeros(len(ACTIONS), dtype=bool)
    if stage != "play":
        return mask

    mask[Action.HIT] = True
    mask[Action.STAND] = True
    mask[Action.DOUBLE] = allow_double and len(player_hand.cards) == 2
    mask[Action.SPLIT] = (
        allow_split
        and player_hand.is_pair
        and len(player_hand.cards) == 2
        and splits_used < max_splits
    )
    mask[Action.SURRENDER] = allow_surrender and len(player_hand.cards) == 2
    if player_hand.doubled:
        mask[:] = False
        mask[Action.STAND] = True
    if player_hand.surrendered:
        mask[:] = False
    return mask


def apply_action_mask(q_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = q_values.copy()
    masked[~mask] = -1e9
    return masked
