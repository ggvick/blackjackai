"""Observation builders for the Blackjack environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .utils import Hand, normalize_feature, true_count, expected_rank_count


@dataclass
class ObservationSpace:
    size: int
    description: List[str]


def dealer_one_hot(rank: int) -> np.ndarray:
    vec = np.zeros(10, dtype=np.float32)
    index = min(rank, 10) - 1
    vec[index] = 1.0
    return vec


def last_action_one_hot(last_action: int | None) -> np.ndarray:
    vec = np.zeros(5, dtype=np.float32)
    if last_action is not None:
        vec[last_action] = 1.0
    return vec


def build_observation(
    dealer_upcard: int,
    player_hand: Hand,
    num_splits_used: int,
    max_splits: int,
    running_count: float,
    decks_remaining: float,
    penetration_progress: float,
    count_10: float,
    count_a: float,
    last_action: int | None,
) -> np.ndarray:
    features: List[np.ndarray] = []
    features.append(dealer_one_hot(dealer_upcard))
    total_norm = normalize_feature(player_hand.total, 0, 31)
    features.append(
        np.array(
            [total_norm, float(player_hand.is_soft), float(player_hand.is_pair)],
            dtype=np.float32,
        )
    )
    splits_norm = normalize_feature(num_splits_used, 0, max_splits or 1)
    features.append(np.array([splits_norm], dtype=np.float32))

    tc = true_count(running_count, max(decks_remaining, 1e-8))
    decks_norm = normalize_feature(decks_remaining, 0.0, 8.0)
    features.append(
        np.array(
            [
                float(tc),
                float(running_count),
                float(decks_norm),
                float(np.clip(penetration_progress, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
    )

    expected = (
        expected_rank_count(int(np.ceil(decks_remaining)))
        if decks_remaining > 0
        else 0.0
    )
    if expected > 0:
        features.append(
            np.array([count_10 / expected, count_a / expected], dtype=np.float32)
        )
    else:
        features.append(np.zeros(2, dtype=np.float32))

    features.append(last_action_one_hot(last_action))

    return np.concatenate(features)


def observation_size() -> ObservationSpace:
    desc: List[str] = []
    desc.extend([f"dealer_{i}" for i in range(10)])
    desc.extend(["player_total_norm", "is_soft", "is_pair", "splits_norm"])
    desc.extend(["true_count", "running_count", "decks_norm", "penetration"])
    desc.extend(["count_10_norm", "count_a_norm"])
    desc.extend(
        [
            "last_action_hit",
            "last_action_stand",
            "last_action_double",
            "last_action_split",
            "last_action_surrender",
        ]
    )
    return ObservationSpace(size=len(desc), description=desc)
