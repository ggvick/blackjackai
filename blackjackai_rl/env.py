"""Advanced Blackjack environment supporting RL training."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .counting import HiLoCounter
from .masking import legal_action_mask
from .observation import ObservationBuilder
from .strategy import basic_strategy
from .utils import bankroll_targets_reached


CARD_SUITS = ("♠", "♥", "♦", "♣")
CARD_RANKS: List[Tuple[str, int]] = [
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


class Phase(enum.Enum):
    BET = "bet"
    PLAY = "play"


@dataclass
class Card:
    rank: str
    value: int
    suit: str

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.rank}{self.suit}"


@dataclass
class Shoe:
    num_decks: int
    rng: np.random.Generator
    cards: List[Card] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.total_cards = self.num_decks * 52
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
    hand_id: int
    split_count: int = 0
    doubled: bool = False
    surrendered: bool = False
    resolved: bool = False
    parent_id: Optional[int] = None

    def clone(self) -> "HandState":
        return HandState(
            cards=list(self.cards),
            bet=self.bet,
            hand_id=self.hand_id,
            split_count=self.split_count,
            doubled=self.doubled,
            surrendered=self.surrendered,
            resolved=self.resolved,
            parent_id=self.parent_id,
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
    shaping_stop_step: int = 60_000
    warmup_basic_alignment_bonus: float = 0.05
    warmup_basic_miss_penalty: float = -0.05
    penetration_reset: bool = True
    bet_actions: int = 8
    bet_sizes: Optional[Sequence[float]] = None
    reward_clip: Optional[float] = 5.0
    seed: Optional[int] = None


class BlackjackEnv:
    """Implements a Blackjack environment tailored for deep RL training."""

    ACTION_STAND = 0
    ACTION_HIT = 1
    ACTION_DOUBLE = 2
    ACTION_SPLIT = 3
    ACTION_SURRENDER = 4

    num_actions = 5

    action_names = {
        ACTION_STAND: "stand",
        ACTION_HIT: "hit",
        ACTION_DOUBLE: "double",
        ACTION_SPLIT: "split",
        ACTION_SURRENDER: "surrender",
    }

    def __init__(self, config: BlackjackEnvConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.counter = HiLoCounter(config.num_decks)
        self.shoe = Shoe(config.num_decks, self.rng)
        self.total_cards = self.shoe.total_cards
        self.observation_builder = ObservationBuilder(
            num_decks=config.num_decks,
            max_splits=config.max_splits,
            min_bet=config.min_bet,
            max_bet=config.max_bet,
            starting_bankroll=config.bankroll,
            bankroll_target=config.bankroll_target,
        )

        self._bet_options = self._compute_bet_options()

        self.episode_id = 0
        self.shoe_id = 0
        self.step_count = 0
        self.episode_step = 0
        self.hands_played = 0
        self.bankroll = config.bankroll
        self.locked_bet = 0.0
        self.last_bet = 0.0
        self.last_action: Optional[int] = None
        self.phase = Phase.BET
        self.active_hands: List[HandState] = []
        self.current_hand_index = 0
        self.dealer_cards: List[Card] = []
        self.dealer_finished = False
        self.cards_drawn = 0
        self.done = False
        self._hand_sequence = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def bet_options(self) -> np.ndarray:
        return np.asarray(self._bet_options, dtype=np.float32)

    @property
    def available_bankroll(self) -> float:
        return max(self.bankroll - self.locked_bet, 0.0)

    @property
    def penetration_progress(self) -> float:
        return (
            0.0
            if self.total_cards == 0
            else float(self.cards_drawn) / float(self.total_cards)
        )

    # ------------------------------------------------------------------
    # Core environment API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, object]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.shoe.rng = self.rng
        self.episode_id += 1
        self.shoe_id = 0
        self.step_count = 0
        self.episode_step = 0
        self.hands_played = 0
        self.bankroll = self.config.bankroll
        self.locked_bet = 0.0
        self.last_bet = 0.0
        self.last_action = None
        self.phase = Phase.BET
        self.active_hands = []
        self.current_hand_index = 0
        self.dealer_cards = []
        self.dealer_finished = False
        self.cards_drawn = 0
        self.done = False
        self._hand_sequence = 0
        self.counter.reset()
        self.shoe.reset()
        self.observation_builder.reset()

        obs = self._build_observation(None)
        info = self._build_info(hand=None)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if self.done:
            obs = self._build_observation(
                None if self.phase is Phase.BET else self._current_hand()
            )
            info = self._build_info(self._current_hand())
            return obs, 0.0, True, info

        self.step_count += 1
        self.episode_step += 1

        if self.phase is Phase.BET:
            return self._handle_bet_action(int(action))
        return self._handle_play_action(int(action))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_bet_options(self) -> List[float]:
        if self.config.bet_sizes is not None:
            unique = sorted({float(b) for b in self.config.bet_sizes if b > 0})
            if not unique:
                raise ValueError("bet_sizes must contain positive values")
            return unique
        levels = np.linspace(
            self.config.min_bet, self.config.max_bet, self.config.bet_actions
        )
        return sorted(float(np.round(level, 2)) for level in np.unique(levels))

    def _current_hand(self) -> Optional[HandState]:
        if 0 <= self.current_hand_index < len(self.active_hands):
            return self.active_hands[self.current_hand_index]
        return None

    def _next_hand_id(self) -> int:
        self._hand_sequence += 1
        return self._hand_sequence

    # Card utilities ----------------------------------------------------
    def _maybe_shuffle_before_round(self) -> None:
        if not self.config.penetration_reset:
            return
        if self.penetration_progress >= self.config.penetration:
            self._reshuffle_shoe()

    def _reshuffle_shoe(self) -> None:
        self.shoe_id += 1
        self.shoe.reset()
        self.counter.reset()
        self.cards_drawn = 0

    def draw_card(self) -> Card:
        if not self.shoe.cards:
            self._reshuffle_shoe()
        card = self.shoe.draw()
        self.cards_drawn += 1
        self.counter.observe(card.rank)
        return card

    @staticmethod
    def hand_total(cards: Sequence[Card]) -> Tuple[int, bool]:
        total = sum(card.value for card in cards)
        aces = sum(1 for card in cards if card.rank == "Ace")
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        soft = aces > 0
        return total, soft

    @staticmethod
    def is_pair(cards: Sequence[Card]) -> Tuple[bool, Optional[str]]:
        if len(cards) != 2:
            return False, None
        a, b = cards
        if a.rank == b.rank:
            return True, a.rank
        if a.value == b.value == 10:
            return True, "10"
        return False, None

    # Observation / info ------------------------------------------------
    def _build_observation(self, hand: Optional[HandState]) -> np.ndarray:
        if self.phase is Phase.BET or hand is None:
            dealer_upcard = self.dealer_cards[0].value if self.dealer_cards else 0
            player_total = 0
            is_soft = False
            is_pair = False
            split_count = 0
            current_bet = 0.0
        else:
            dealer_upcard = self.dealer_cards[0].value if self.dealer_cards else 0
            player_total, is_soft = self.hand_total(hand.cards)
            is_pair, _ = self.is_pair(hand.cards[:2])
            split_count = hand.split_count
            current_bet = hand.bet
        obs = self.observation_builder.build(
            dealer_upcard=dealer_upcard,
            player_total=player_total,
            is_soft=is_soft,
            is_pair=is_pair,
            split_count=split_count,
            counter=self.counter,
            cards_drawn=self.cards_drawn,
            total_cards=self.total_cards,
            last_action=self.last_action,
            bankroll=self.bankroll,
            locked_bet=self.locked_bet,
            current_bet=current_bet,
            last_bet=self.last_bet,
            num_active_hands=len(self.active_hands),
            episode_step=self.episode_step,
            hands_played=self.hands_played,
            phase_is_bet=self.phase is Phase.BET,
        )
        return obs

    def _base_info(self) -> Dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "shoe_id": self.shoe_id,
            "bankroll": float(self.bankroll),
            "penetration": float(self.penetration_progress),
            "phase": self.phase.value,
            "needs_bet": self.phase is Phase.BET,
            "bet_options": self.bet_options.copy(),
        }

    def _build_info(self, hand: Optional[HandState]) -> Dict[str, object]:
        info = self._base_info()
        if hand is not None:
            total, _ = self.hand_total(hand.cards)
            info.update(
                {
                    "hand_id": hand.hand_id,
                    "player_total": total,
                    "hand_cards": tuple(card.rank for card in hand.cards),
                    "current_bet": float(hand.bet),
                }
            )
            legal = self.valid_actions(hand)
            info["legal_actions"] = list(legal)
            info["action_mask"] = legal_action_mask(legal, self.num_actions)
        else:
            info["legal_actions"] = []
            info["action_mask"] = np.ones(self.num_actions, dtype=np.float32)
        return info

    # Valid actions -----------------------------------------------------
    def valid_actions(self, hand: HandState) -> List[int]:
        total, _ = self.hand_total(hand.cards)
        is_pair, _ = self.is_pair(hand.cards)
        legal = [self.ACTION_STAND, self.ACTION_HIT]
        if (
            self.config.allow_double
            and len(hand.cards) == 2
            and self.available_bankroll >= hand.bet
        ):
            legal.append(self.ACTION_DOUBLE)
        if (
            self.config.allow_split
            and len(hand.cards) == 2
            and is_pair
            and hand.split_count < self.config.max_splits
            and self.available_bankroll >= hand.bet
        ):
            legal.append(self.ACTION_SPLIT)
        if (
            self.config.allow_surrender
            and len(hand.cards) == 2
            and not hand.surrendered
            and total in {15, 16}
        ):
            legal.append(self.ACTION_SURRENDER)
        return sorted(set(legal))

    # Phase handlers ----------------------------------------------------
    def _handle_bet_action(
        self, bet_action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if (
            bankroll_targets_reached(
                self.bankroll,
                self.config.bankroll_stop_loss,
                self.config.bankroll_target,
            )
            or self.bankroll <= 0.0
        ):
            self.done = True
            obs = self._build_observation(None)
            info = self._build_info(None)
            info["outcome"] = "terminal"
            return obs, 0.0, True, info

        bet_index = int(np.clip(bet_action, 0, len(self._bet_options) - 1))
        requested_bet = self._bet_options[bet_index]
        bet = float(min(max(requested_bet, self.config.min_bet), self.config.max_bet))
        if self.bankroll < self.config.min_bet:
            bet = float(self.bankroll)
        bet = float(np.clip(bet, 0.0, self.bankroll))
        if bet <= 0.0:
            self.done = True
            obs = self._build_observation(None)
            info = self._build_info(None)
            info["outcome"] = "bankroll_empty"
            return obs, 0.0, True, info

        self._maybe_shuffle_before_round()

        self.last_bet = bet
        self.locked_bet = bet
        self.dealer_cards = [self.draw_card(), self.draw_card()]
        player_cards = [self.draw_card(), self.draw_card()]
        self.active_hands = [
            HandState(cards=player_cards, bet=bet, hand_id=self._next_hand_id())
        ]
        self.current_hand_index = 0
        self.dealer_finished = False
        self.phase = Phase.PLAY

        hand = self.active_hands[0]
        reward, info, immediate_done = self._check_naturals(hand)
        if info.get("hand_complete"):
            obs, info_final = self._advance_after_hand(hand, info)
            reward = self._clip_reward(reward)
            return obs, reward, immediate_done or self.done, info_final

        obs = self._build_observation(hand)
        info = self._build_info(hand)
        info["outcome"] = "deal"
        return obs, 0.0, False, info

    def _handle_play_action(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        hand = self._current_hand()
        if hand is None:
            self.phase = Phase.BET
            obs = self._build_observation(None)
            info = self._build_info(None)
            return obs, 0.0, self.done, info

        legal = self.valid_actions(hand)
        if action not in legal:
            action = legal[0]
        self.last_action = action

        total, usable_ace = self.hand_total(hand.cards)
        is_pair, pair_rank = self.is_pair(hand.cards)
        dealer_upcard = self.dealer_cards[0].value if self.dealer_cards else 0
        recommended = basic_strategy(
            total,
            usable_ace,
            pair_rank,
            dealer_upcard,
            self.ACTION_DOUBLE in legal,
            self.ACTION_SPLIT in legal,
            self.config.allow_surrender and len(hand.cards) == 2,
        )
        recommended_action = self._strategy_to_action(recommended.action)

        reward = 0.0
        info: Dict[str, object]
        hand_complete = False

        if action == self.ACTION_HIT:
            hand.cards.append(self.draw_card())
            total, _ = self.hand_total(hand.cards)
            if total > 21:
                reward, info = self._resolve_hand(hand, outcome="player_bust")
                hand_complete = True
            else:
                obs = self._build_observation(hand)
                info = self._build_info(hand)
                info.update({"outcome": "continue"})
                shaping = self._shaping_bonus(action, recommended_action)
                reward = self._clip_reward(shaping)
                return obs, reward, False, info
        elif action == self.ACTION_DOUBLE:
            reward, info = self._handle_double(hand)
            hand_complete = True
        elif action == self.ACTION_SPLIT:
            obs, info = self._handle_split(hand)
            shaping = self._shaping_bonus(action, recommended_action)
            reward = self._clip_reward(shaping)
            return obs, reward, False, info
        elif action == self.ACTION_SURRENDER:
            hand.surrendered = True
            reward, info = self._resolve_hand(hand, outcome="surrender")
            hand_complete = True
        else:  # stand
            reward, info = self._resolve_hand(hand, outcome="stand")
            hand_complete = True

        shaping = self._shaping_bonus(action, recommended_action)
        reward += shaping
        reward = self._clip_reward(reward)

        obs, final_info = self._advance_after_hand(hand, info)
        done = self.done
        if hand_complete:
            final_info["hand_complete"] = True
        return obs, reward, done, final_info

    def _shaping_bonus(self, action: int, recommended_action: int) -> float:
        if not self.config.reward_shaping:
            return 0.0
        if self.step_count > self.config.shaping_stop_step:
            return 0.0
        return (
            self.config.warmup_basic_alignment_bonus
            if action == recommended_action
            else self.config.warmup_basic_miss_penalty
        )

    def _handle_double(self, hand: HandState) -> Tuple[float, Dict[str, object]]:
        additional_bet = hand.bet
        if self.available_bankroll < additional_bet:
            return self._resolve_hand(hand, outcome="stand")
        hand.doubled = True
        hand.bet += additional_bet
        self.locked_bet += additional_bet
        hand.cards.append(self.draw_card())
        return self._resolve_hand(hand, outcome="double")

    def _handle_split(self, hand: HandState) -> Tuple[np.ndarray, Dict[str, object]]:
        card_a, card_b = hand.cards
        additional_bet = hand.bet
        if self.available_bankroll < additional_bet:
            obs = self._build_observation(hand)
            info = self._build_info(hand)
            info.update({"outcome": "split_denied"})
            return obs, info
        self.locked_bet += additional_bet
        new_hand_a = HandState(
            cards=[card_a, self.draw_card()],
            bet=hand.bet,
            hand_id=self._next_hand_id(),
            split_count=hand.split_count + 1,
            parent_id=hand.hand_id,
        )
        new_hand_b = HandState(
            cards=[card_b, self.draw_card()],
            bet=hand.bet,
            hand_id=self._next_hand_id(),
            split_count=hand.split_count + 1,
            parent_id=hand.hand_id,
        )
        self.active_hands[self.current_hand_index] = new_hand_a
        self.active_hands.insert(self.current_hand_index + 1, new_hand_b)
        obs = self._build_observation(new_hand_a)
        info = self._build_info(new_hand_a)
        info.update({"outcome": "split"})
        return obs, info

    def _resolve_hand(
        self, hand: HandState, outcome: str
    ) -> Tuple[float, Dict[str, object]]:
        hand.resolved = True
        delta = 0.0
        if hand.surrendered:
            delta = -0.5 * hand.bet
        else:
            player_total, _ = self.hand_total(hand.cards)
            if player_total > 21:
                delta = -hand.bet
                outcome = "player_bust"
            else:
                dealer_total = self._dealer_total()
                if dealer_total > 21 or player_total > dealer_total:
                    delta = hand.bet
                    outcome = "win"
                elif player_total < dealer_total:
                    delta = -hand.bet
                    outcome = "loss"
                else:
                    delta = 0.0
                    outcome = "push"
        self._release_bet(hand.bet)
        self.bankroll += delta
        self._update_terminal()
        info = self._build_info(hand)
        info.update({"outcome": outcome, "delta": delta})
        return delta, info

    def _dealer_total(self) -> int:
        if self.dealer_finished:
            total, _ = self.hand_total(self.dealer_cards)
            return total
        total, usable = self.hand_total(self.dealer_cards)
        while total < 17 or (total == 17 and usable and self.config.hit_soft_17):
            self.dealer_cards.append(self.draw_card())
            total, usable = self.hand_total(self.dealer_cards)
        self.dealer_finished = True
        return total

    def _release_bet(self, amount: float) -> None:
        self.locked_bet = max(self.locked_bet - amount, 0.0)

    def _check_naturals(self, hand: HandState) -> Tuple[float, Dict[str, object], bool]:
        player_blackjack = self.is_blackjack(hand.cards)
        dealer_blackjack = self.is_blackjack(self.dealer_cards)
        if not player_blackjack and not dealer_blackjack:
            info = self._build_info(hand)
            return 0.0, info, False

        delta = 0.0
        outcome = "push"
        if player_blackjack and dealer_blackjack:
            delta = 0.0
            outcome = "push"
        elif player_blackjack:
            delta = hand.bet * self.config.natural_payout
            outcome = "blackjack"
        else:
            delta = -hand.bet
            outcome = "dealer_blackjack"

        hand.resolved = True
        self._release_bet(hand.bet)
        self.bankroll += delta
        self._update_terminal()
        info = self._build_info(hand)
        info.update({"outcome": outcome, "hand_complete": True, "delta": delta})
        return delta, info, self.done

    def is_blackjack(self, cards: Sequence[Card]) -> bool:
        total, _ = self.hand_total(cards)
        return total == 21 and len(cards) == 2

    def _advance_after_hand(
        self, hand: HandState, info: Dict[str, object]
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        if hand in self.active_hands:
            self.active_hands[self.current_hand_index] = hand
        if all(h.resolved for h in self.active_hands):
            self.hands_played += 1
            self.active_hands = []
            self.current_hand_index = 0
            self.dealer_cards = []
            self.dealer_finished = False
            self.locked_bet = 0.0
            self.phase = Phase.BET
            obs = self._build_observation(None)
            final_info = self._merge_info(self._build_info(None), info)
            return obs, final_info

        # move to next unresolved hand
        for idx, candidate in enumerate(self.active_hands):
            if not candidate.resolved:
                self.current_hand_index = idx
                break
        self.phase = Phase.PLAY
        next_hand = self.active_hands[self.current_hand_index]
        obs = self._build_observation(next_hand)
        final_info = self._merge_info(self._build_info(next_hand), info)
        return obs, final_info

    def _merge_info(
        self, base: Dict[str, object], extra: Dict[str, object]
    ) -> Dict[str, object]:
        for key, value in extra.items():
            if key in {"legal_actions", "action_mask", "phase", "bet_options"}:
                continue
            base[key] = value
        return base

    def _update_terminal(self) -> None:
        if bankroll_targets_reached(
            self.bankroll, self.config.bankroll_stop_loss, self.config.bankroll_target
        ):
            self.done = True

    def _clip_reward(self, reward: float) -> float:
        if self.config.reward_clip is None:
            return reward
        return float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

    def _strategy_to_action(self, name: str) -> int:
        reverse = {v: k for k, v in self.action_names.items()}
        return reverse.get(name, self.ACTION_STAND)


class VectorizedBlackjackEnv:
    """Batch many Blackjack environments for efficient experience collection."""

    def __init__(self, num_envs: int, config: BlackjackEnvConfig):
        self.envs = [BlackjackEnv(config) for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        observations = []
        infos: List[Dict[str, object]] = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), infos

    def step(
        self, actions: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
        obs_batch: List[np.ndarray] = []
        reward_batch: List[float] = []
        done_batch: List[float] = []
        info_batch: List[Dict[str, object]] = []
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(int(action))
            if done:
                reset_obs, reset_info = env.reset()
                info["reset_info"] = reset_info
                obs = reset_obs
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
