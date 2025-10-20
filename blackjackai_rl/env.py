"""Advanced Blackjack environment supporting RL training."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .counting import HiLoCounter
from .strategy import basic_strategy
from .utils import bankroll_targets_reached


CARD_SUITS = ("♠", "♥", "♦", "♣")
CARD_RANKS = [
    ("Ace", 11),
    ("2", 2),
    ("3", 3),
    ("4", 4),
    ("5", 5),
    ("6", 6),
    ("7", 7),
    ("8", 8),
    ("9", 9),
    ("10", 10),
    ("Jack", 10),
    ("Queen", 10),
    ("King", 10),
]


@dataclass
class Card:
    rank: str
    value: int
    suit: str


@dataclass
class Shoe:
    num_decks: int
    rng: np.random.Generator
    cards: List[Card] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cards = [
            Card(rank, value, suit)
            for _ in range(self.num_decks)
            for rank, value in CARD_RANKS
            for suit in CARD_SUITS
        ]
        self.rng.shuffle(self.cards)

    def draw(self) -> Card:
        if not self.cards:
            raise RuntimeError("Shoe is empty; cannot draw")
        return self.cards.pop()

    @property
    def cards_remaining(self) -> int:
        return len(self.cards)


@dataclass
class HandState:
    cards: List[Card]
    bet: float
    doubled: bool = False
    surrendered: bool = False
    resolved: bool = False
    split_count: int = 0
    origin_pair_rank: Optional[str] = None

    def clone(self) -> "HandState":
        return HandState(
            cards=list(self.cards),
            bet=self.bet,
            doubled=self.doubled,
            surrendered=self.surrendered,
            resolved=self.resolved,
            split_count=self.split_count,
            origin_pair_rank=self.origin_pair_rank,
        )


@dataclass
class BlackjackEnvConfig:
    num_decks: int = 6
    penetration: float = 0.8
    natural_payout: float = 1.5
    hit_soft_17: bool = False
    min_bet: float = 1.0
    max_bet: float = 8.0
    bankroll: float = 100.0
    bankroll_stop_loss: float = 0.0
    bankroll_target: float = 200.0
    allow_surrender: bool = True
    allow_double: bool = True
    allow_split: bool = True
    max_splits: int = 1
    reward_shaping: bool = True
    shaping_stop_step: int = 50_000
    warmup_basic_alignment_bonus: float = 0.05
    warmup_basic_miss_penalty: float = -0.05
    penetration_reset: bool = True
    counting_bet_units: Tuple[int, int, int, int] = (1, 2, 4, 8)
    true_count_thresholds: Tuple[int, int, int] = (1, 2, 3)
    seed: Optional[int] = None


class BlackjackEnv:
    """Implements the full Blackjack ruleset for reinforcement learning."""

    ACTION_STAND = 0
    ACTION_HIT = 1
    ACTION_DOUBLE = 2
    ACTION_SPLIT = 3
    ACTION_SURRENDER = 4

    action_names = {
        ACTION_STAND: "stand",
        ACTION_HIT: "hit",
        ACTION_DOUBLE: "double",
        ACTION_SPLIT: "split",
        ACTION_SURRENDER: "surrender",
    }

    def _strategy_to_action(self, name: str) -> int:
        reverse = {v: k for k, v in self.action_names.items()}
        return reverse.get(name, self.ACTION_STAND)

    def __init__(self, config: BlackjackEnvConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.counter = HiLoCounter(config.num_decks)
        self.shoe = Shoe(config.num_decks, self.rng)
        self.cards_drawn = 0
        self.shuffle_at = int(self.config.num_decks * 52 * self.config.penetration)
        self.current_step = 0

        self.bankroll = config.bankroll
        self.active_hands: List[HandState] = []
        self.current_hand_index = 0
        self.dealer_cards: List[Card] = []
        self.done = False
        self.last_info: Dict[str, float | str] = {}
        self.last_reward = 0.0

    # ------------------------------------------------------------------
    # Helpers for card/hand evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def hand_total(cards: List[Card]) -> Tuple[int, bool]:
        total = sum(card.value for card in cards)
        ace_count = sum(1 for card in cards if card.rank == "Ace")
        # adjust for aces valued as 1 instead of 11
        soft = False
        while total > 21 and ace_count:
            total -= 10
            ace_count -= 1
        if any(card.rank == "Ace" for card in cards) and total <= 21:
            # determine if there is a usable ace (counted as 11)
            soft = any(card.rank == "Ace" for card in cards) and total <= 21 and any(
                card.rank == "Ace" for card in cards
            )
        usable_ace = soft and any(card.rank == "Ace" for card in cards)
        return total, usable_ace

    @staticmethod
    def is_blackjack(cards: List[Card]) -> bool:
        total, _ = BlackjackEnv.hand_total(cards)
        return total == 21 and len(cards) == 2

    @staticmethod
    def is_pair(cards: List[Card]) -> Tuple[bool, Optional[str]]:
        if len(cards) != 2:
            return False, None
        if cards[0].rank == cards[1].rank:
            return True, cards[0].rank
        if cards[0].value == cards[1].value == 10:
            return True, "10"
        return False, None

    def _maybe_reshuffle(self) -> None:
        if not self.config.penetration_reset:
            return
        if self.cards_drawn >= self.shuffle_at or not self.shoe.cards:
            self.shoe.reset()
            self.counter.reset()
            self.cards_drawn = 0

    def draw_card(self) -> Card:
        self._maybe_reshuffle()
        card = self.shoe.draw()
        self.cards_drawn += 1
        self.counter.observe(card.rank)
        return card

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------
    def reset(self) -> Tuple[np.ndarray, Dict[str, float | str]]:
        self.bankroll = self.config.bankroll
        self.counter.reset()
        self.shoe.reset()
        self.cards_drawn = 0
        self.shuffle_at = int(self.config.num_decks * 52 * self.config.penetration)
        self.current_step = 0
        self.done = False
        self.active_hands = []
        self.dealer_cards = []
        obs, info = self._start_round()
        return obs, info

    def _start_round(self) -> Tuple[np.ndarray, Dict[str, float | str]]:
        self.current_hand_index = 0
        self.active_hands = []
        self.dealer_cards = []
        true_count = self.counter.true_count
        bet = self._bet_from_count(true_count)
        bet = min(bet, self.bankroll)
        if bet < self.config.min_bet and self.bankroll > 0:
            bet = min(self.config.min_bet, self.bankroll)
        if self.bankroll <= 0 or bet <= 0:
            self.done = True
            return self._get_obs(None), self._build_info("bankroll_empty")

        player_cards = [self.draw_card(), self.draw_card()]
        dealer_cards = [self.draw_card(), self.draw_card()]
        self.dealer_cards = dealer_cards

        hand = HandState(cards=player_cards, bet=bet)
        pair, pair_rank = self.is_pair(player_cards)
        if pair:
            hand.origin_pair_rank = pair_rank
        self.active_hands = [hand]

        immediate_reward, info = self._check_naturals(hand)
        if info.get("hand_complete"):
            self._apply_delta(immediate_reward)
            if self.done:
                return self._get_obs(None), info
            obs = self._prepare_next_hand()
            return obs, info

        obs = self._get_obs(self.active_hands[self.current_hand_index])
        info.setdefault("legal_actions", self.valid_actions(hand))
        return obs, info

    # ------------------------------------------------------------------
    # Betting helpers
    # ------------------------------------------------------------------
    def _bet_from_count(self, true_count: float) -> float:
        thresholds = self.config.true_count_thresholds
        spreads = self.config.counting_bet_units
        unit = spreads[0]
        for threshold, spread in zip(thresholds, spreads[1:]):
            if true_count >= threshold:
                unit = spread
        bet = unit * self.config.min_bet
        return min(bet, self.config.max_bet)

    # ------------------------------------------------------------------
    # Natural blackjack handling
    # ------------------------------------------------------------------
    def _check_naturals(self, hand: HandState) -> Tuple[float, Dict[str, float | str]]:
        player_blackjack = self.is_blackjack(hand.cards)
        dealer_blackjack = self.is_blackjack(self.dealer_cards)
        if not player_blackjack and not dealer_blackjack:
            return 0.0, self._build_info("play", hand)

        delta = 0.0
        outcome = "push"
        if player_blackjack and dealer_blackjack:
            delta = -0.1  # push reward penalty
        elif player_blackjack:
            win_amount = hand.bet * (self.config.natural_payout)
            delta = win_amount
            outcome = "blackjack_win"
        else:
            loss_amount = hand.bet
            delta = -loss_amount
            outcome = "dealer_blackjack"

        info = self._build_info(outcome, hand)
        info.update({
            "hand_complete": True,
            "delta": delta,
            "bankroll": self.bankroll + delta,
        })

        self.done = bankroll_targets_reached(
            self.bankroll, self.config.bankroll_stop_loss, self.config.bankroll_target
        ) or self.shoe.cards_remaining <= max(1, int((1 - self.config.penetration) * self.config.num_decks * 52))
        return delta, info

    # ------------------------------------------------------------------
    def _prepare_next_hand(self) -> np.ndarray:
        # remove resolved hands
        remaining = [hand for hand in self.active_hands if not hand.resolved]
        self.active_hands = remaining
        if self.active_hands:
            self.current_hand_index = 0
            return self._get_obs(self.active_hands[0])

        if self.done or bankroll_targets_reached(
            self.bankroll, self.config.bankroll_stop_loss, self.config.bankroll_target
        ):
            self.done = True
            return self._get_obs(None)

        if self.shoe.cards_remaining < (1 - self.config.penetration) * self.config.num_decks * 52:
            self.done = True
            return self._get_obs(None)

        obs, _info = self._start_round()
        return obs

    # ------------------------------------------------------------------
    # Observation / info generation
    # ------------------------------------------------------------------
    def _get_obs(self, hand: Optional[HandState]) -> np.ndarray:
        if hand is None:
            return np.zeros(12, dtype=np.float32)
        total, usable_ace = self.hand_total(hand.cards)
        is_pair, pair_rank = self.is_pair(hand.cards)
        dealer_upcard = self.dealer_cards[0].value if self.dealer_cards else 0
        true_count = self.counter.true_count
        running_count = self.counter.running_count
        decks_remaining = self.counter.decks_remaining / max(self.config.num_decks, 1e-9)
        bet_fraction = hand.bet / max(self.config.max_bet, 1e-9)
        bankroll_ratio = self.bankroll / max(self.config.bankroll_target, 1e-9)
        step_norm = math.tanh(self.current_step / 5000.0)
        return np.array(
            [
                total / 21.0,
                float(usable_ace),
                float(is_pair),
                dealer_upcard / 11.0,
                true_count / 10.0,
                running_count / 20.0,
                bet_fraction,
                bankroll_ratio,
                decks_remaining,
                float(hand.doubled),
                step_norm,
                float(hand.split_count),
            ],
            dtype=np.float32,
        )

    def _build_info(self, outcome: str, hand: Optional[HandState] = None) -> Dict[str, float | str]:
        dealer_total, _ = self.hand_total(self.dealer_cards)
        info: Dict[str, float | str] = {
            "outcome": outcome,
            "dealer_total": float(dealer_total),
            "true_count": float(self.counter.true_count),
            "running_count": float(self.counter.running_count),
            "bankroll": float(self.bankroll),
        }
        if hand is not None:
            player_total, _ = self.hand_total(hand.cards)
            info.update(
                {
                    "player_total": float(player_total),
                    "bet": float(hand.bet),
                    "hand_cards": tuple(card.rank for card in hand.cards),
                }
            )
        return info

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------
    def valid_actions(self, hand: Optional[HandState] = None) -> List[int]:
        if hand is None:
            return [self.ACTION_STAND]
        total, _ = self.hand_total(hand.cards)
        is_pair, pair_rank = self.is_pair(hand.cards)
        actions = [self.ACTION_STAND, self.ACTION_HIT]
        bankroll_available = self.bankroll
        if self.config.allow_double and len(hand.cards) == 2 and bankroll_available >= hand.bet:
            actions.append(self.ACTION_DOUBLE)
        if (
            self.config.allow_split
            and len(hand.cards) == 2
            and is_pair
            and hand.split_count < self.config.max_splits
            and bankroll_available >= hand.bet
        ):
            actions.append(self.ACTION_SPLIT)
        if self.config.allow_surrender and len(hand.cards) == 2 and total in {15, 16}:
            actions.append(self.ACTION_SURRENDER)
        return actions

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float | str]]:
        if self.done:
            return self._get_obs(None), 0.0, True, self._build_info("episode_done")

        hand = self.active_hands[self.current_hand_index]
        legal_actions = self.valid_actions(hand)

        # Basic strategy guidance for reward shaping
        total, usable_ace = self.hand_total(hand.cards)
        is_pair, pair_rank = self.is_pair(hand.cards)
        dealer_upcard = self.dealer_cards[0].value if self.dealer_cards else 0
        recommended = basic_strategy(
            total,
            usable_ace,
            pair_rank,
            dealer_upcard,
            self.ACTION_DOUBLE in legal_actions,
            self.ACTION_SPLIT in legal_actions,
            self.config.allow_surrender and len(hand.cards) == 2,
        )
        recommended_action = self._strategy_to_action(recommended.action)

        if action not in legal_actions:
            action = self.ACTION_STAND
        self.current_step += 1
        reward = 0.0
        info: Dict[str, float | str] = {}

        if action == self.ACTION_HIT:
            hand.cards.append(self.draw_card())
            total, _ = self.hand_total(hand.cards)
            if total > 21:
                reward, info = self._resolve_hand(hand, outcome="bust")
            else:
                obs = self._get_obs(hand)
                info = self._build_info("continue", hand)
                info.update({"legal_actions": legal_actions})
                return obs, reward, False, info
        elif action == self.ACTION_DOUBLE:
            hand.doubled = True
            hand.bet = min(hand.bet * 2, self.config.max_bet)
            hand.cards.append(self.draw_card())
            reward, info = self._resolve_hand(hand, outcome="double")
        elif action == self.ACTION_SPLIT:
            reward, info = self._split_hand(hand)
            obs = self._get_obs(self.active_hands[self.current_hand_index])
            info.update({"legal_actions": self.valid_actions(self.active_hands[self.current_hand_index])})
            return obs, reward, False, info
        elif action == self.ACTION_SURRENDER:
            hand.surrendered = True
            reward, info = self._resolve_hand(hand, outcome="surrender")
        else:  # stand
            reward, info = self._resolve_hand(hand, outcome="stand")

        shaping_bonus = 0.0
        if self.config.reward_shaping and self.current_step <= self.config.shaping_stop_step:
            if action == recommended_action:
                shaping_bonus = self.config.warmup_basic_alignment_bonus
            else:
                shaping_bonus = self.config.warmup_basic_miss_penalty
        reward += shaping_bonus

        terminal_bonus = 0.0
        prev_done = self.done
        self._apply_delta(reward - shaping_bonus)  # bankroll should track financial delta only
        if not prev_done and self.done:
            if self.bankroll >= self.config.bankroll_target:
                terminal_bonus = 2.0
            elif self.bankroll <= self.config.bankroll_stop_loss:
                terminal_bonus = -2.0
        reward += terminal_bonus

        obs = self._prepare_next_hand()
        done = self.done
        info.setdefault(
            "legal_actions",
            self.valid_actions(self.active_hands[0]) if self.active_hands else [self.ACTION_STAND],
        )
        info.update(
            {
                "recommended_action": recommended.action,
                "shaping_bonus": shaping_bonus,
                "terminal_bonus": terminal_bonus,
                "bankroll": self.bankroll,
            }
        )
        return obs, reward, done, info

    # ------------------------------------------------------------------
    def _split_hand(self, hand: HandState) -> Tuple[float, Dict[str, float | str]]:
        card_a, card_b = hand.cards
        hand_a = HandState(cards=[card_a, self.draw_card()], bet=hand.bet, split_count=hand.split_count + 1)
        hand_b = HandState(cards=[card_b, self.draw_card()], bet=hand.bet, split_count=hand.split_count + 1)
        is_pair, pair_rank = self.is_pair([card_a, card_b])
        if is_pair:
            hand_a.origin_pair_rank = pair_rank
            hand_b.origin_pair_rank = pair_rank
        # replace current hand with two new hands
        self.active_hands.pop(self.current_hand_index)
        self.active_hands.insert(self.current_hand_index, hand_b)
        self.active_hands.insert(self.current_hand_index, hand_a)
        reward = -abs(hand.bet) * 0.01  # tiny penalty to avoid degenerate loops
        info = self._build_info("split", hand)
        return reward, info

    def _dealer_play(self) -> Tuple[int, bool]:
        total, usable = self.hand_total(self.dealer_cards)
        while total < 17 or (total == 17 and usable and self.config.hit_soft_17):
            self.dealer_cards.append(self.draw_card())
            total, usable = self.hand_total(self.dealer_cards)
        return total, usable

    def _resolve_hand(self, hand: HandState, outcome: str) -> Tuple[float, Dict[str, float | str]]:
        hand.resolved = True
        if hand.surrendered:
            delta = -0.5 * hand.bet
            info = self._build_info("surrender", hand)
            info.update({"hand_complete": True, "delta": delta})
            return delta, info

        player_total, _ = self.hand_total(hand.cards)
        if player_total > 21:
            delta = -hand.bet
            info = self._build_info("player_bust", hand)
            info.update({"hand_complete": True, "delta": delta})
            return delta, info

        dealer_total, _ = self._dealer_play()
        delta = 0.0
        result = "push"
        if dealer_total > 21 or player_total > dealer_total:
            delta = hand.bet
            result = "win"
        elif player_total < dealer_total:
            delta = -hand.bet
            result = "loss"
        else:
            delta = -0.1  # push penalty
        info = self._build_info(result, hand)
        info.update({"hand_complete": True, "delta": delta})
        return delta, info

    def _apply_delta(self, delta: float) -> None:
        self.bankroll += delta
        if bankroll_targets_reached(
            self.bankroll, self.config.bankroll_stop_loss, self.config.bankroll_target
        ):
            self.done = True
        if self.shoe.cards_remaining <= max(
            1, int((1 - self.config.penetration) * self.config.num_decks * 52)
        ):
            self.done = True


class VectorizedBlackjackEnv:
    """Batch many Blackjack environments for efficient experience collection."""

    def __init__(self, num_envs: int, config: BlackjackEnvConfig):
        self.envs = [BlackjackEnv(config) for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self) -> Tuple[np.ndarray, List[Dict[str, float | str]]]:
        observations = []
        infos: List[Dict[str, float | str]] = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), infos

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float | str]]]:
        obs_batch, reward_batch, done_batch, info_batch = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(int(action))
            if done:
                obs_reset, info_reset = env.reset()
                obs = obs_reset
                info["reset"] = info_reset
            obs_batch.append(obs)
            reward_batch.append(reward)
            done_batch.append(float(done))
            info_batch.append(info)
        return (
            np.stack(obs_batch),
            np.asarray(reward_batch, dtype=np.float32),
            np.asarray(done_batch, dtype=np.float32),
            info_batch,
        )


__all__ = [
    "Card",
    "Shoe",
    "HandState",
    "BlackjackEnvConfig",
    "BlackjackEnv",
    "VectorizedBlackjackEnv",
]
