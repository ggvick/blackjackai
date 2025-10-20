"""Blackjack environment implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .count_helpers import CountState
from .masking import ACTIONS, Action, legal_action_mask
from .observation import ObservationSpace, build_observation, observation_size
from .utils import Hand, build_shoe, penetration_reached


@dataclass
class EnvConfig:
    num_decks: int = 6
    penetration: float = 0.75
    bankroll_start: float = 100.0
    bankroll_target: float = 200.0
    min_bet: float = 5.0
    max_bet: float = 100.0
    bet_levels: int = 8
    natural_payout: float = 1.5
    hit_soft_17: bool = True
    allow_surrender: bool = True
    allow_double: bool = True
    allow_split: bool = True
    max_splits: int = 3
    seed: int | None = None

    def bet_amount(self, bet_action: int) -> float:
        bet_action = int(np.clip(bet_action, 0, self.bet_levels - 1))
        if self.bet_levels == 1:
            return self.min_bet
        step = (self.max_bet - self.min_bet) / (self.bet_levels - 1)
        return self.min_bet + bet_action * step


@dataclass
class EnvState:
    bankroll: float
    shoe: List[int]
    shoe_initial_count: int
    dealer_hand: Optional[Hand] = None
    player_hands: List[Hand] = field(default_factory=list)
    hand_bets: List[float] = field(default_factory=list)
    current_hand_index: int = 0
    stage: str = "bet"
    running_count: float = 0.0
    num_splits_used: int = 0
    last_action: Optional[int] = None
    count_state: CountState = field(default_factory=CountState)
    round_number: int = 0
    steps: int = 0


class BlackjackEnv:
    """Environment modelling multi-hand blackjack with bankroll objective."""

    observation_space: ObservationSpace

    def __init__(self, config: EnvConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.observation_space = observation_size()
        self.state = self._initial_state()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
        self.state = self._initial_state()
        return self._current_observation()

    def step(
        self, action: int | Dict[str, int] | Tuple[int, int]
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        state = self.state
        reward = 0.0
        info: Dict = {}

        if state.stage == "bet":
            bet_action = self._extract_bet_action(action)
            bet = self.config.bet_amount(bet_action)
            self._start_round(bet)
            info["bet"] = bet
            observation = self._current_observation()
            done = self._episode_done()
            return observation, reward, done, info

        play_action = self._extract_play_action(action)
        info["action_name"] = ACTIONS[play_action]
        reward, round_complete = self._apply_player_action(play_action)
        done = False
        if round_complete:
            round_reward = self._resolve_round()
            reward += round_reward
            done = self._episode_done()
            if not done:
                self.state.stage = "bet"
                observation = self._current_observation()
            else:
                observation = self._terminal_observation()
            info["round_complete"] = True
            info["bankroll"] = self.state.bankroll
        else:
            observation = self._current_observation()
        self.state.steps += 1
        info["mask"] = self._current_action_mask()
        return observation, reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _initial_state(self) -> EnvState:
        shoe = build_shoe(self.config.num_decks, self.rng)
        return EnvState(
            bankroll=self.config.bankroll_start,
            shoe=shoe,
            shoe_initial_count=len(shoe),
        )

    def _episode_done(self) -> bool:
        return (
            self.state.bankroll <= 0
            or self.state.bankroll >= self.config.bankroll_target
        )

    # Observation -------------------------------------------------------
    def _current_observation(self) -> np.ndarray:
        state = self.state
        if state.stage == "bet" or not state.player_hands:
            return np.zeros(self.observation_space.size, dtype=np.float32)

        player_hand = state.player_hands[state.current_hand_index]
        dealer_upcard = state.dealer_hand.cards[0]
        decks_remaining = state.count_state.decks_remaining(self.config.num_decks)
        penetration = state.count_state.penetration(self.config.num_decks)
        obs = build_observation(
            dealer_upcard=dealer_upcard,
            player_hand=player_hand,
            num_splits_used=state.num_splits_used,
            max_splits=self.config.max_splits,
            running_count=state.count_state.running_count,
            decks_remaining=decks_remaining,
            penetration_progress=penetration,
            count_10=state.count_state.count_10,
            count_a=state.count_state.count_a,
            last_action=state.last_action,
        )
        return obs

    def _terminal_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.size, dtype=np.float32)

    # Betting -----------------------------------------------------------
    def _extract_bet_action(
        self, action: int | Dict[str, int] | Tuple[int, int]
    ) -> int:
        if isinstance(action, dict):
            return int(action.get("bet", 0))
        if isinstance(action, tuple):
            return int(action[0])
        return int(action)

    def _extract_play_action(
        self, action: int | Dict[str, int] | Tuple[int, int]
    ) -> int:
        if isinstance(action, dict):
            result = int(action.get("play", 0))
        if isinstance(action, tuple):
            result = int(action[1]) if len(action) > 1 else int(action[0])
        else:
            result = int(action)
        return max(0, min(result, len(ACTIONS) - 1))

    def _draw_card(self) -> int:
        state = self.state
        if not state.shoe:
            state.shoe = build_shoe(self.config.num_decks, self.rng)
            state.shoe_initial_count = len(state.shoe)
            state.count_state = CountState()
        card = state.shoe.pop()
        state.count_state.update([card])
        return card

    def _start_round(self, bet: float) -> None:
        state = self.state
        if penetration_reached(
            state.shoe_initial_count, len(state.shoe), self.config.penetration
        ):
            state.shoe = build_shoe(self.config.num_decks, self.rng)
            state.shoe_initial_count = len(state.shoe)
            state.count_state = CountState()

        state.dealer_hand = Hand(cards=[])
        state.player_hands = [Hand(cards=[])]
        state.hand_bets = [bet]
        state.current_hand_index = 0
        state.stage = "play"
        state.num_splits_used = 0
        state.last_action = None
        state.round_number += 1

        # deal cards: player-dealer-player-dealer
        order = [
            state.player_hands[0],
            state.dealer_hand,
            state.player_hands[0],
            state.dealer_hand,
        ]
        for hand in order:
            card = self._draw_card()
            hand.add_card(card)

    # Player actions ----------------------------------------------------
    def _apply_player_action(self, action: int) -> Tuple[float, bool]:
        state = self.state
        hand = state.player_hands[state.current_hand_index]
        mask = self._current_action_mask()
        if not mask[action]:
            raise ValueError(f"Illegal action {action}")

        reward = 0.0
        if action == Action.HIT:
            card = self._draw_card()
            hand.add_card(card)
            state.last_action = action
            if hand.is_bust():
                hand.resolved = True
        elif action == Action.STAND:
            hand.resolved = True
            state.last_action = action
        elif action == Action.DOUBLE:
            card = self._draw_card()
            hand.add_card(card)
            hand.doubled = True
            hand.resolved = True
            state.hand_bets[state.current_hand_index] *= 2
            state.last_action = action
        elif action == Action.SPLIT:
            state.num_splits_used += 1
            card_a, card_b = hand.cards
            new_hand_a = Hand(cards=[card_a])
            new_hand_b = Hand(cards=[card_b])
            hand.cards = [card_a]
            state.hand_bets[state.current_hand_index] = state.hand_bets[
                state.current_hand_index
            ]
            state.player_hands[state.current_hand_index] = new_hand_a
            state.player_hands.insert(state.current_hand_index + 1, new_hand_b)
            state.hand_bets.insert(
                state.current_hand_index + 1, state.hand_bets[state.current_hand_index]
            )
            for idx in (state.current_hand_index, state.current_hand_index + 1):
                card = self._draw_card()
                state.player_hands[idx].add_card(card)
            state.last_action = action
        elif action == Action.SURRENDER:
            hand.surrendered = True
            hand.resolved = True
            state.last_action = action
        else:  # pragma: no cover - unreachable due to mask check
            raise ValueError("Unknown action")

        round_complete = self._advance_hand_pointer()
        return reward, round_complete

    def _hand_bet(self, hand: Hand) -> float:
        return self.state.hand_bets[self.state.current_hand_index]

    def _advance_hand_pointer(self) -> bool:
        state = self.state
        while (
            state.current_hand_index < len(state.player_hands)
            and state.player_hands[state.current_hand_index].resolved
        ):
            state.current_hand_index += 1
        if state.current_hand_index >= len(state.player_hands):
            return True
        return False

    # Resolution --------------------------------------------------------
    def _resolve_round(self) -> float:
        state = self.state
        dealer_hand = state.dealer_hand
        assert dealer_hand is not None
        # Dealer draws if needed
        while dealer_hand.total < 17 or (
            dealer_hand.total == 17 and self.config.hit_soft_17 and dealer_hand.is_soft
        ):
            card = self._draw_card()
            dealer_hand.add_card(card)
        dealer_total = dealer_hand.total
        dealer_bust = dealer_total > 21

        total_reward = 0.0
        for hand, bet in zip(state.player_hands, state.hand_bets):
            if hand.surrendered:
                total_reward += -0.5 * bet
                continue
            if hand.is_bust():
                total_reward += -bet
                continue
            if hand.is_blackjack and dealer_hand.is_blackjack:
                total_reward += 0.0
                continue
            if hand.is_blackjack:
                total_reward += self.config.natural_payout * bet
                continue
            if dealer_hand.is_blackjack:
                total_reward += -bet
                continue
            if dealer_bust:
                total_reward += bet
                continue
            player_total = hand.total
            if player_total > dealer_total:
                total_reward += bet
            elif player_total < dealer_total:
                total_reward += -bet
            else:
                total_reward += 0.0
        state.bankroll += total_reward
        state.stage = "bet"
        state.player_hands = []
        state.hand_bets = []
        state.dealer_hand = None
        state.current_hand_index = 0
        state.last_action = None
        return total_reward

    # Mask --------------------------------------------------------------
    def _current_action_mask(self) -> np.ndarray:
        state = self.state
        if state.stage != "play" or not state.player_hands:
            return np.zeros(len(ACTIONS), dtype=bool)
        hand = state.player_hands[state.current_hand_index]
        return legal_action_mask(
            player_hand=hand,
            stage=state.stage,
            allow_double=self.config.allow_double,
            allow_split=self.config.allow_split,
            allow_surrender=self.config.allow_surrender,
            max_splits=self.config.max_splits,
            splits_used=state.num_splits_used,
        )

    # Convenience -------------------------------------------------------
    def available_actions(self) -> np.ndarray:
        return self._current_action_mask()

    def clone_state(self) -> EnvState:
        return self.state


class VectorizedBlackjackEnv:
    """Simple vectorized wrapper executing multiple environments in lockstep."""

    def __init__(self, config: EnvConfig, num_envs: int):
        self.envs = [BlackjackEnv(config) for _ in range(num_envs)]

    def reset(self) -> np.ndarray:
        observations = [env.reset() for env in self.envs]
        return np.stack(observations)

    def step(
        self, bet_actions: Sequence[int], play_actions: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        observations: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict] = []
        for env, bet_action, play_action in zip(self.envs, bet_actions, play_actions):
            if env.state.stage == "bet":
                obs, reward, done, info = env.step({"bet": bet_action})
                if done:
                    observations.append(obs)
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)
                    continue
            obs, reward, done, info = env.step(play_action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:  # pragma: no cover - compatibility
        pass
