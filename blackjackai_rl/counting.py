"""Card counting utilities for Blackjack."""
from __future__ import annotations

from dataclasses import dataclass

RANKS_PLUS_ONE = {"2", "3", "4", "5", "6"}
RANKS_MINUS_ONE = {"10", "Jack", "Queen", "King", "Ace"}


@dataclass
class HiLoCounter:
    """Classic Hi-Lo running/true count tracker."""

    num_decks: int
    running_count: float = 0.0
    cards_remaining: int | None = None

    def __post_init__(self) -> None:
        self.total_cards = self.num_decks * 52
        self.reset()

    def reset(self) -> None:
        self.running_count = 0.0
        self.cards_remaining = self.total_cards

    def observe(self, rank: str) -> None:
        if self.cards_remaining is None or self.cards_remaining <= 0:
            return
        if rank in RANKS_PLUS_ONE:
            self.running_count += 1.0
        elif rank in RANKS_MINUS_ONE:
            self.running_count -= 1.0
        self.cards_remaining -= 1

    @property
    def decks_remaining(self) -> float:
        if not self.cards_remaining:
            return 0.0
        return max(self.cards_remaining / 52.0, 1e-3)

    @property
    def true_count(self) -> float:
        if not self.cards_remaining:
            return float(self.running_count)
        return self.running_count / self.decks_remaining


__all__ = ["HiLoCounter"]
