"""Blackjack reinforcement learning environment with card counting support."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from blackjack.models.card import Card
from blackjack.models.hand import DealerHand, GamblerHand
from blackjack.models.shoe import Shoe


@dataclass
class HiLoCounter:
    """Simple Hi-Lo card counting implementation."""

    num_decks: int
    running_count: int = 0
    cards_remaining: int = 0

    def __post_init__(self) -> None:
        self.total_cards = self.num_decks * 52
        self.reset()

    def reset(self) -> None:
        """Reset the running count for a freshly shuffled shoe."""
        self.running_count = 0
        self.cards_remaining = self.total_cards

    @staticmethod
    def _count_value(card: Card) -> int:
        """Return the Hi-Lo contribution for a dealt card."""
        if card.name in {"2", "3", "4", "5", "6"}:
            return 1
        if card.name in {"10", "Jack", "Queen", "King", "Ace"}:
            return -1
        return 0

    def observe(self, card: Card) -> None:
        """Update the running and true counts after a card is dealt."""
        if self.cards_remaining:
            self.running_count += self._count_value(card)
            self.cards_remaining -= 1

    def true_count(self) -> float:
        """Return the true count (running count / decks remaining)."""
        if self.cards_remaining == 0:
            return float(self.running_count)
        decks_remaining = max(self.cards_remaining / 52.0, 0.25)
        return self.running_count / decks_remaining


@dataclass
class CountingPolicy:
    """Map true count values to bet multipliers."""

    min_bet: float
    max_bet: float
    ramp: Optional[Dict[int, float]] = None

    def __post_init__(self) -> None:
        if self.max_bet < self.min_bet:
            raise ValueError("max_bet must be >= min_bet")
        if self.ramp is None:
            # Default bet ramp inspired by common Hi-Lo spreads.
            self.ramp = {1: 2, 2: 4, 3: 6, 4: 8}

    def bet_for_true_count(self, true_count: float) -> float:
        """Return the wager amount for a given true count."""
        bet = self.min_bet
        for threshold, multiplier in sorted(self.ramp.items()):
            if true_count >= threshold:
                bet = min(self.min_bet * multiplier, self.max_bet)
        return float(bet)


class BlackjackEnv:
    """Lightweight environment tailored for tabular reinforcement learning."""

    ACTION_STAND = 0
    ACTION_HIT = 1
    ACTION_DOUBLE = 2

    def __init__(
        self,
        num_decks: int = 6,
        penetration: float = 0.75,
        natural_payout: float = 1.5,
        hit_soft_17: bool = False,
        min_bet: float = 5.0,
        max_bet: float = 100.0,
        counting_policy: Optional[CountingPolicy] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not 0 < penetration <= 1:
            raise ValueError("penetration must be in (0, 1]")
        self.num_decks = num_decks
        self.penetration = penetration
        self.natural_payout = natural_payout
        self.hit_soft_17 = hit_soft_17
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.counting_policy = counting_policy or CountingPolicy(min_bet, max_bet)
        self.random = random.Random(seed)

        self.shoe = Shoe(num_decks)
        self.counter = HiLoCounter(num_decks)
        self.total_cards = num_decks * 52
        self.shuffle_at = int(self.total_cards * self.penetration)
        self.cards_drawn = 0
        self.needs_shuffle = True

        self.player_hand: GamblerHand
        self.dealer_hand: DealerHand
        self.current_bet: float = self.min_bet
        self.can_double: bool = True
        self.round_complete: bool = False
        self.last_info: Dict[str, float] = {}
        self.last_reward: float = 0.0

    # ------------------------------------------------------------------
    # Deck / shoe helpers
    # ------------------------------------------------------------------
    def _reshuffle(self) -> None:
        self.shoe.reset_card_pile()
        self.counter.reset()
        self.cards_drawn = 0
        self.needs_shuffle = False

    def _draw_card(self) -> Card:
        """Deal a card, updating the counter and shoe state."""
        if self.needs_shuffle or self.cards_drawn >= self.shuffle_at:
            self._reshuffle()
        card = self.shoe.deal_card()
        self.counter.observe(card)
        self.cards_drawn += 1
        return card

    # ------------------------------------------------------------------
    # Hand utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _hand_value(cards: Iterable[Card]) -> Tuple[int, bool]:
        total = 0
        aces = 0
        for card in cards:
            if card.name == "Ace":
                total += 1
                aces += 1
            else:
                total += card.value
        usable_ace = False
        if aces and total + 10 <= 21:
            total += 10
            usable_ace = True
        return total, usable_ace

    @staticmethod
    def _card_display_value(card: Card) -> int:
        if card.name == "Ace":
            return 11
        return card.value

    def _initial_deal(self) -> None:
        self.player_hand = GamblerHand(cards=[self._draw_card(), self._draw_card()])
        self.dealer_hand = DealerHand(cards=[self._draw_card(), self._draw_card()])
        self.can_double = True
        self.round_complete = False
        self.last_reward = 0.0
        self.last_info = {}

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------
    @property
    def action_space(self) -> Tuple[int, ...]:
        return (self.ACTION_STAND, self.ACTION_HIT, self.ACTION_DOUBLE)

    def valid_actions(self) -> Tuple[int, ...]:
        actions = [self.ACTION_STAND, self.ACTION_HIT]
        if self.can_double:
            actions.append(self.ACTION_DOUBLE)
        return tuple(actions)

    def sample_action(self) -> int:
        return self.random.choice(self.valid_actions())

    def reset(self) -> Tuple[int, int, int, int, int]:
        """Start a new round and return the initial observation."""
        self.current_bet = self.counting_policy.bet_for_true_count(self.counter.true_count())
        self._initial_deal()
        self._handle_naturals()
        return self._get_state()

    def _handle_naturals(self) -> None:
        player_total, _ = self._hand_value(self.player_hand.cards)
        dealer_total, _ = self._hand_value(self.dealer_hand.cards)
        player_blackjack = player_total == 21 and len(self.player_hand.cards) == 2
        dealer_blackjack = dealer_total == 21 and len(self.dealer_hand.cards) == 2
        if player_blackjack or dealer_blackjack:
            if player_blackjack and dealer_blackjack:
                outcome = "push"
                reward = 0.0
            elif player_blackjack:
                outcome = "win"
                reward = self.current_bet * self.natural_payout
            else:
                outcome = "loss"
                reward = -self.current_bet
            self.round_complete = True
            self.last_reward = reward
            self.last_info = {
                "outcome": outcome,
                "player_total": player_total,
                "dealer_total": dealer_total,
                "bet": self.current_bet,
                "true_count": self.counter.true_count(),
            }

    def _get_state(self) -> Tuple[int, int, int, int, int]:
        player_total, usable_ace = self._hand_value(self.player_hand.cards)
        dealer_upcard = self._card_display_value(self.dealer_hand.up_card())
        true_count_bucket = int(max(-6, min(6, round(self.counter.true_count()))))
        return (
            max(4, min(player_total, 31)),
            dealer_upcard,
            int(usable_ace),
            true_count_bucket,
            int(self.can_double),
        )

    def step(self, action: int) -> Tuple[Tuple[int, int, int, int, int], float, bool, Dict[str, float]]:
        """Execute a player action and return environment feedback."""
        if self.round_complete:
            return self._get_state(), 0.0, True, self.last_info

        if action not in self.valid_actions():
            action = self.ACTION_STAND

        if action == self.ACTION_DOUBLE:
            self.current_bet = min(self.current_bet * 2, self.max_bet)
            self.player_hand.cards.append(self._draw_card())
            reward, info = self._resolve_round(player_finished=True)
            return self._terminal_transition(reward, info)

        if action == self.ACTION_HIT:
            self.player_hand.cards.append(self._draw_card())
            self.can_double = False
            player_total, _ = self._hand_value(self.player_hand.cards)
            if player_total > 21:
                reward = -self.current_bet
                info = self._build_info("loss")
                return self._terminal_transition(reward, info)
            return self._get_state(), 0.0, False, self._build_info("pending")

        # Stand
        reward, info = self._resolve_round(player_finished=True)
        return self._terminal_transition(reward, info)

    # ------------------------------------------------------------------
    # Round resolution helpers
    # ------------------------------------------------------------------
    def _terminal_transition(
        self, reward: float, info: Dict[str, float]
    ) -> Tuple[Tuple[int, int, int, int, int], float, bool, Dict[str, float]]:
        self.round_complete = True
        self.last_reward = reward
        self.last_info = info
        return self._get_state(), reward, True, info

    def _resolve_round(self, player_finished: bool) -> Tuple[float, Dict[str, float]]:
        player_total, _ = self._hand_value(self.player_hand.cards)
        if player_total > 21:
            return -self.current_bet, self._build_info("loss")

        dealer_total, dealer_usable_ace = self._hand_value(self.dealer_hand.cards)
        while True:
            soft = dealer_usable_ace and dealer_total == 17
            should_hit = dealer_total < 17 or (soft and self.hit_soft_17)
            if not should_hit:
                break
            self.dealer_hand.cards.append(self._draw_card())
            dealer_total, dealer_usable_ace = self._hand_value(self.dealer_hand.cards)

        if dealer_total > 21:
            return self.current_bet, self._build_info("win", dealer_total)
        if player_total > dealer_total:
            return self.current_bet, self._build_info("win", dealer_total)
        if player_total < dealer_total:
            return -self.current_bet, self._build_info("loss", dealer_total)
        return 0.0, self._build_info("push", dealer_total)

    def _build_info(self, outcome: str, dealer_total: Optional[int] = None) -> Dict[str, float]:
        player_total, _ = self._hand_value(self.player_hand.cards)
        if dealer_total is None:
            dealer_total, _ = self._hand_value(self.dealer_hand.cards)
        return {
            "outcome": outcome,
            "player_total": player_total,
            "dealer_total": dealer_total,
            "bet": self.current_bet,
            "true_count": self.counter.true_count(),
        }
