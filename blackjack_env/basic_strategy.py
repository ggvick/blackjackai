"""Simplified basic strategy implementation."""

from __future__ import annotations

from dataclasses import dataclass

from .masking import Action
from .utils import Hand


@dataclass
class BasicStrategyPolicy:
    """Implements a conservative basic strategy policy."""

    allow_double: bool = True
    allow_split: bool = True
    allow_surrender: bool = True

    def act(self, dealer_upcard: int, hand: Hand, mask) -> int:
        total = hand.total
        is_soft = hand.is_soft
        if mask[Action.SURRENDER] and total == 16 and dealer_upcard in (9, 10, 1):
            return Action.SURRENDER
        if mask[Action.SPLIT] and hand.is_pair:
            if hand.cards[0] in (1, 8):
                return Action.SPLIT
            if hand.cards[0] in (9,) and dealer_upcard not in (7, 10, 1):
                return Action.SPLIT
        if is_soft:
            if total <= 17:
                return Action.HIT
            if total == 18:
                if dealer_upcard in (2, 7, 8):
                    return Action.STAND
                if mask[Action.DOUBLE] and dealer_upcard in (3, 4, 5, 6):
                    return Action.DOUBLE
                return Action.HIT
            return Action.STAND
        if total <= 11:
            if total == 11 and mask[Action.DOUBLE]:
                return Action.DOUBLE
            if total == 10 and mask[Action.DOUBLE] and dealer_upcard not in (10, 1):
                return Action.DOUBLE
            return Action.HIT
        if total == 12:
            if dealer_upcard in (4, 5, 6):
                return Action.STAND
            return Action.HIT
        if 13 <= total <= 16:
            if dealer_upcard >= 7:
                return Action.HIT
            return Action.STAND
        return Action.STAND
