"""Counting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .utils import card_value


@dataclass
class CountState:
    running_count: float = 0.0
    dealt_cards: int = 0
    count_10: float = 0.0
    count_a: float = 0.0

    def update(self, cards: Iterable[int]) -> None:
        for card in cards:
            value = card_value(card)
            self.dealt_cards += 1
            if value in (10,):
                self.count_10 += 1
            if value == 1:
                self.count_a += 1
            if value in (10, 1):
                self.running_count -= 1
            elif value in (2, 3, 4, 5, 6):
                self.running_count += 1

    def decks_remaining(self, num_decks: int) -> float:
        total_cards = 52 * num_decks
        remaining_cards = max(total_cards - self.dealt_cards, 1)
        return remaining_cards / 52

    def penetration(self, num_decks: int) -> float:
        total_cards = 52 * num_decks
        return min(self.dealt_cards / total_cards, 1.0)
