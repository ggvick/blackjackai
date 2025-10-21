"""Utility helpers for the Blackjack environment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

CARD_RANKS: Sequence[int] = tuple(range(1, 14))
CARD_VALUES = {**{i: min(i, 10) for i in CARD_RANKS}}


def card_value(card: int) -> int:
    """Return the blackjack value of a card rank."""
    return CARD_VALUES[card]


@dataclass
class Hand:
    """Utility structure representing a blackjack hand."""

    cards: List[int]
    doubled: bool = False
    surrendered: bool = False
    resolved: bool = False

    def add_card(self, card: int) -> None:
        self.cards.append(card)

    @property
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.total == 21 and self.is_soft

    @property
    def is_pair(self) -> bool:
        return len(self.cards) == 2 and card_value(self.cards[0]) == card_value(
            self.cards[1]
        )

    @property
    def is_soft(self) -> bool:
        return any(card == 1 for card in self.cards) and self.total <= 21

    @property
    def total(self) -> int:
        total = sum(card_value(card) for card in self.cards)
        aces = sum(1 for c in self.cards if c == 1)
        while aces > 0 and total + 10 <= 21:
            total += 10
            aces -= 1
        return total

    def is_bust(self) -> bool:
        return self.total > 21

    def copy(self) -> "Hand":
        return Hand(
            cards=list(self.cards),
            doubled=self.doubled,
            surrendered=self.surrendered,
            resolved=self.resolved,
        )


def draw_cards(rng: random.Random, shoe: List[int], num_cards: int) -> List[int]:
    return [shoe.pop() for _ in range(num_cards)]


def build_shoe(num_decks: int, rng: random.Random) -> List[int]:
    shoe = list(CARD_RANKS) * 4 * num_decks
    rng.shuffle(shoe)
    return shoe


def penetration_reached(
    initial_count: int, current_count: int, penetration: float
) -> bool:
    dealt = initial_count - current_count
    return dealt / initial_count >= penetration


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass


def running_count_update(running_count: float, cards: Iterable[int]) -> float:
    """Apply Hi-Lo count update."""
    for card in cards:
        value = card_value(card)
        if value in (1, 10):
            running_count -= 1
        elif value in (2, 3, 4, 5, 6):
            running_count += 1
    return running_count


def true_count(running_count: float, decks_remaining: float) -> float:
    if decks_remaining <= 0:
        return 0.0
    return running_count / decks_remaining


def normalize_feature(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0
    value = (value - min_value) / (max_value - min_value)
    return float(np.clip(value, 0.0, 1.0))


def expected_rank_count(num_decks: int) -> float:
    return 4 * num_decks / 13


def format_cards(cards: Sequence[int]) -> str:
    return "[" + ", ".join(str(card) for card in cards) + "]"
