"""Deterministic basic-strategy policy tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActionName = Literal["hit", "stand", "double", "split", "surrender"]


@dataclass(frozen=True)
class BasicStrategyDecision:
    action: ActionName
    rationale: str


SURRENDER_RULES = {
    (16, False): {9, 10, 11},
    (15, False): {10},
}


def _hard_total_action(total: int, dealer: int, can_double: bool) -> ActionName:
    if total <= 8:
        return "hit"
    if total == 9:
        return "double" if can_double and 3 <= dealer <= 6 else "hit"
    if total == 10:
        return "double" if can_double and 2 <= dealer <= 9 else "hit"
    if total == 11:
        return "double" if can_double and dealer <= 10 else "hit"
    if total == 12:
        return "stand" if 4 <= dealer <= 6 else "hit"
    if 13 <= total <= 16:
        return "stand" if 2 <= dealer <= 6 else "hit"
    return "stand"


def _soft_total_action(total: int, dealer: int, can_double: bool) -> ActionName:
    if total <= 17:
        return "hit"
    if total == 18:
        if 2 <= dealer <= 6:
            return "double" if can_double else "stand"
        if dealer in {7, 8}:
            return "stand"
        return "hit"
    if total == 19:
        return "double" if can_double and dealer in {6} else "stand"
    if total >= 20:
        return "stand"
    return "hit"


PAIR_RULES = {
    "Ace": {"split": {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
    "10": {"stand": set(range(2, 12))},
    "9": {"split": {2, 3, 4, 5, 6, 8, 9}, "stand": {7, 10, 11}},
    "8": {"split": set(range(2, 12))},
    "7": {"split": {2, 3, 4, 5, 6, 7}},
    "6": {"split": {2, 3, 4, 5, 6}},
    "5": {"double": {2, 3, 4, 5, 6, 7, 8, 9}},
    "4": {"split": {5, 6}},
    "3": {"split": {2, 3, 4, 5, 6, 7}},
    "2": {"split": {2, 3, 4, 5, 6, 7}},
}


def basic_strategy(
    total: int,
    is_soft: bool,
    pair_rank: str | None,
    dealer_upcard: int,
    can_double: bool,
    can_split: bool,
    allow_surrender: bool,
) -> BasicStrategyDecision:
    """Return the suggested action under standard S17 multi-deck rules."""

    if allow_surrender and not is_soft and pair_rank is None:
        surrender_dealers = SURRENDER_RULES.get((total, False))
        if surrender_dealers and dealer_upcard in surrender_dealers:
            return BasicStrategyDecision("surrender", "Hard total surrender rule")

    if can_split and pair_rank is not None:
        rules = PAIR_RULES.get(pair_rank)
        if rules:
            if "split" in rules and dealer_upcard in rules["split"]:
                return BasicStrategyDecision("split", "Pair splitting table")
            if "double" in rules and dealer_upcard in rules["double"]:
                if can_double:
                    return BasicStrategyDecision("double", "Treat pair of fives as double")
                return BasicStrategyDecision("hit", "Pair double fallback")
            if "stand" in rules and dealer_upcard in rules["stand"]:
                return BasicStrategyDecision("stand", "Pair standing table")

    if is_soft:
        action = _soft_total_action(total, dealer_upcard, can_double)
        rationale = "Soft total table"
    else:
        action = _hard_total_action(total, dealer_upcard, can_double)
        rationale = "Hard total table"

    return BasicStrategyDecision(action, rationale)


__all__ = ["BasicStrategyDecision", "basic_strategy"]
