"""Lightweight smoke tests for the Blackjack RL package."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .agents import TabularConfig, TabularQLearningAgent
from .device import detect_torch_device
from .env import Card, HandState, BlackjackEnv, BlackjackEnvConfig


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str = ""


def test_env_reset_and_step() -> TestResult:
    config = BlackjackEnvConfig(num_decks=1, penetration=0.5, reward_shaping=False)
    env = BlackjackEnv(config)
    obs, info = env.reset()
    try:
        assert obs.shape == (12,)
        assert "legal_actions" in info
        action = info["legal_actions"][0]
        obs2, reward, done, info2 = env.step(action)
        assert obs2.shape == (12,)
        assert isinstance(reward, float)
        assert "bankroll" in info2
    except AssertionError as exc:
        return TestResult("env_reset_step", False, str(exc))
    return TestResult("env_reset_step", True)


def test_shoe_penetration() -> TestResult:
    config = BlackjackEnvConfig(num_decks=1, penetration=0.25, reward_shaping=False)
    env = BlackjackEnv(config)
    env.reset()
    for _ in range(env.shuffle_at + 5):
        env.draw_card()
    # After reshuffle the running count should be near zero and cards_drawn small
    passed = env.cards_drawn <= env.shuffle_at and abs(env.counter.running_count) <= 5
    return TestResult("shoe_penetration", passed, "penetration logic failed" if not passed else "")


def _make_card(rank: str, value: int) -> Card:
    return Card(rank=rank, value=value, suit="♠")


def test_reward_function() -> TestResult:
    config = BlackjackEnvConfig(num_decks=1, penetration=0.5, reward_shaping=False, allow_split=False)
    env = BlackjackEnv(config)
    env.reset()
    # Win scenario
    env.dealer_cards = [_make_card("9", 9), _make_card("7", 7)]
    player_hand = HandState(cards=[_make_card("10", 10), _make_card("8", 8)], bet=2.0)
    reward, info = env._resolve_hand(player_hand, outcome="stand")
    if not np.isclose(reward, 2.0):
        return TestResult("reward_win", False, f"expected 2.0, got {reward}")
    # Push
    env.dealer_cards = [_make_card("10", 10), _make_card("7", 7)]
    player_hand = HandState(cards=[_make_card("10", 10), _make_card("7", 7)], bet=2.0)
    reward, info = env._resolve_hand(player_hand, outcome="stand")
    if not np.isclose(reward, -0.1):
        return TestResult("reward_push", False, f"expected -0.1, got {reward}")
    # Doubled loss
    env.dealer_cards = [_make_card("10", 10), _make_card("9", 9)]
    player_hand = HandState(cards=[_make_card("10", 10), _make_card("6", 6)], bet=4.0)
    reward, info = env._resolve_hand(player_hand, outcome="stand")
    if not np.isclose(reward, -4.0):
        return TestResult("reward_loss", False, f"expected -4.0, got {reward}")
    return TestResult("reward_math", True)


def test_device_selection() -> TestResult:
    device = detect_torch_device()
    if not device.device:
        return TestResult("device_detection", False, "device not detected")
    return TestResult("device_detection", True, device.device)


def test_tabular_overfit() -> TestResult:
    agent = TabularQLearningAgent(num_actions=5, config=TabularConfig(lr=0.5, epsilon_start=0.0, epsilon_final=0.0))
    state = np.zeros(12, dtype=np.float32)
    next_state = np.zeros(12, dtype=np.float32)
    legal = [0, 1]
    agent.update(state, 0, reward=1.0, next_obs=next_state, done=False, legal_next=legal)
    q_value = agent.q_table[agent._discretize(state)][0]
    if q_value <= 0.0:
        return TestResult("tabular_overfit", False, f"q_value not updated: {q_value}")
    return TestResult("tabular_overfit", True)


def run_all_tests() -> List[TestResult]:
    tests = [
        test_env_reset_and_step,
        test_shoe_penetration,
        test_reward_function,
        test_device_selection,
        test_tabular_overfit,
    ]
    results = [test() for test in tests]
    return results


__all__ = ["TestResult", "run_all_tests"]
