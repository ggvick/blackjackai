"""Observation builder for Blackjack RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .counting import HiLoCounter


@dataclass
class ObservationBuilder:
    """Construct rich feature vectors for the Blackjack environment.

    The builder centralises feature engineering so both training and evaluation
    code consume identical observations.  The feature ordering is intentionally
    fixed and documented here for use in agents and tests.

    Feature layout (length = 36):
        0-9   : Dealer up-card one-hot encoding for ranks 2-10, Ace
        10    : Player total / 21
        11    : Player has a soft total (usable Ace)
        12    : Player hand is a pair
        13    : Number of splits used / max_splits
        14    : Running count (Hi-Lo) clipped to [-20, 20] then / 20
        15    : True count clipped to [-10, 10] then / 10
        16    : Decks remaining clipped to [0, num_decks] then / num_decks
        17    : Penetration progress (cards dealt / total cards)
        18-22 : Last action one-hot (hit/stand/double/split/surrender)
        23    : Bankroll / bankroll_target (clipped to [0, 2])
        24    : Bankroll / starting_bankroll (clipped to [0, 2])
        25    : Current bet / max_bet
        26    : Available bankroll (after locking bets) / max_bet
        27    : Active hands / (max_splits + 1)
        28    : Episode step normalised via tanh(step / 5000)
        29    : Hands played normalised via tanh(hands / 1000)
        30    : Phase indicator (bet)
        31    : Phase indicator (play)
        32    : True count positive indicator
        33    : Running count positive indicator
        34    : Remaining penetration (1 - penetration progress)
        35    : Last bet / max_bet
    """

    num_decks: int
    max_splits: int
    min_bet: float
    max_bet: float
    starting_bankroll: float
    bankroll_target: float

    observation_dim: int = 36

    def reset(self) -> None:  # pragma: no cover - placeholder for symmetry
        """Reset any internal state if required."""

    @staticmethod
    def _dealer_one_hot(upcard_value: int) -> np.ndarray:
        mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}
        one_hot = np.zeros(10, dtype=np.float32)
        if upcard_value in mapping:
            one_hot[mapping[upcard_value]] = 1.0
        return one_hot

    def _normalise_count(self, value: float, limit: float) -> float:
        clipped = float(np.clip(value, -limit, limit))
        return clipped / limit if limit > 0 else 0.0

    def _normalise_positive(self, value: float, limit: float) -> float:
        clipped = float(np.clip(value, 0.0, limit))
        return clipped / limit if limit > 0 else 0.0

    def build(
        self,
        *,
        dealer_upcard: int,
        player_total: int,
        is_soft: bool,
        is_pair: bool,
        split_count: int,
        counter: HiLoCounter,
        cards_drawn: int,
        total_cards: int,
        last_action: Optional[int],
        bankroll: float,
        locked_bet: float,
        current_bet: float,
        last_bet: float,
        num_active_hands: int,
        episode_step: int,
        hands_played: int,
        phase_is_bet: bool,
    ) -> np.ndarray:
        decks_remaining = counter.decks_remaining
        penetration_progress = (
            float(cards_drawn) / float(total_cards) if total_cards else 0.0
        )
        remaining_penetration = 1.0 - penetration_progress

        features: list[float] = []
        features.extend(self._dealer_one_hot(dealer_upcard))
        features.append(float(player_total) / 21.0)
        features.append(1.0 if is_soft else 0.0)
        features.append(1.0 if is_pair else 0.0)
        split_norm_denom = max(self.max_splits, 1)
        features.append(float(split_count) / float(split_norm_denom))
        features.append(self._normalise_count(counter.running_count, 20.0))
        features.append(self._normalise_count(counter.true_count, 10.0))
        features.append(
            self._normalise_positive(decks_remaining, float(self.num_decks))
        )
        features.append(float(np.clip(penetration_progress, 0.0, 1.0)))

        last_action_one_hot = np.zeros(5, dtype=np.float32)
        if last_action is not None and 0 <= last_action < 5:
            last_action_one_hot[last_action] = 1.0
        features.extend(last_action_one_hot.tolist())

        target = max(self.bankroll_target, 1e-6)
        starting = max(self.starting_bankroll, 1e-6)
        features.append(float(np.clip(bankroll / target, 0.0, 2.0)))
        features.append(float(np.clip(bankroll / starting, 0.0, 2.0)))
        features.append(float(np.clip(current_bet / max(self.max_bet, 1e-6), 0.0, 1.0)))
        available = float(np.clip(bankroll - locked_bet, 0.0, self.max_bet))
        features.append(float(np.clip(available / max(self.max_bet, 1e-6), 0.0, 1.0)))
        features.append(
            float(np.clip(num_active_hands / float(self.max_splits + 1), 0.0, 1.0))
        )
        features.append(float(np.tanh(episode_step / 5000.0)))
        features.append(float(np.tanh(hands_played / 1000.0)))
        features.append(1.0 if phase_is_bet else 0.0)
        features.append(0.0 if phase_is_bet else 1.0)
        features.append(1.0 if counter.true_count > 0 else 0.0)
        features.append(1.0 if counter.running_count > 0 else 0.0)
        features.append(float(np.clip(remaining_penetration, 0.0, 1.0)))
        features.append(float(np.clip(last_bet / max(self.max_bet, 1e-6), 0.0, 1.0)))

        obs = np.asarray(features, dtype=np.float32)
        assert obs.shape[0] == self.observation_dim, obs.shape
        return obs


__all__ = ["ObservationBuilder"]
