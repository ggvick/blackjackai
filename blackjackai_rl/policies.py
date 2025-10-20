"""Baseline policies used for evaluation benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .env import BlackjackEnv
from .strategy import basic_strategy


class EvaluationPolicy:
    def select_bet(self, env: BlackjackEnv, info: dict) -> int:
        raise NotImplementedError

    def select_play(self, env: BlackjackEnv, info: dict) -> int:
        raise NotImplementedError

    def act(self, env: BlackjackEnv, obs, info: dict) -> int:
        if info.get("needs_bet", info.get("phase") == "bet"):
            return self.select_bet(env, info)
        return self.select_play(env, info)


class BasicStrategyPolicy(EvaluationPolicy):
    def __init__(self, bet_index: int = 0) -> None:
        self.bet_index = bet_index

    def select_bet(self, env: BlackjackEnv, info: dict) -> int:
        return min(self.bet_index, env.config.bet_actions - 1)

    def select_play(self, env: BlackjackEnv, info: dict) -> int:
        hand = env.active_hands[env.current_hand_index]
        legal = env.valid_actions(hand)
        total, usable_ace = env.hand_total(hand.cards)
        is_pair, pair_rank = env.is_pair(hand.cards)
        dealer_upcard = env.dealer_cards[0].value if env.dealer_cards else 0
        decision = basic_strategy(
            total,
            usable_ace,
            pair_rank,
            dealer_upcard,
            env.ACTION_DOUBLE in legal,
            env.ACTION_SPLIT in legal,
            env.config.allow_surrender and len(hand.cards) == 2,
        )
        action = env._strategy_to_action(decision.action)
        if action in legal:
            return action
        if env.ACTION_HIT in legal:
            return env.ACTION_HIT
        return env.ACTION_STAND


@dataclass
class CountBettingSchedule:
    thresholds: Sequence[float]
    bet_indices: Sequence[int]


class CountBettingPolicy(BasicStrategyPolicy):
    """Basic strategy with a simple count-driven betting ramp."""

    def __init__(self, schedule: CountBettingSchedule) -> None:
        super().__init__(bet_index=0)
        self.schedule = schedule

    def select_bet(self, env: BlackjackEnv, info: dict) -> int:
        true_count = env.counter.true_count
        index = 0
        for threshold, bet_index in zip(
            self.schedule.thresholds, self.schedule.bet_indices
        ):
            if true_count >= threshold:
                index = bet_index
        return min(index, env.config.bet_actions - 1)


__all__ = [
    "EvaluationPolicy",
    "BasicStrategyPolicy",
    "CountBettingPolicy",
    "CountBettingSchedule",
]
