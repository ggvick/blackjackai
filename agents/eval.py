"""Evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from blackjack_env.env import BlackjackEnv


@dataclass
class EvalResult:
    ev_per_hand: float
    ev_per_100: float
    ci_low: float
    ci_high: float
    win_rate: float
    loss_rate: float
    push_rate: float


def evaluate_policy(env: BlackjackEnv, policy, num_hands: int = 1000) -> EvalResult:
    rewards: List[float] = []
    outcomes = {"win": 0, "loss": 0, "push": 0}
    env.reset()
    hand_counter = 0
    while hand_counter < num_hands:
        if env.state.stage == "bet":
            env.step({"bet": 0})
            continue
        mask = env.available_actions()
        action = policy(
            env.state.dealer_hand.cards[0],
            env.state.player_hands[env.state.current_hand_index],
            mask,
        )
        obs, reward, done, info = env.step(action)
        if info.get("round_complete"):
            if reward > 0:
                outcomes["win"] += 1
            elif reward < 0:
                outcomes["loss"] += 1
            else:
                outcomes["push"] += 1
            rewards.append(reward)
            hand_counter += 1
        if done:
            env.reset()
    rewards_arr = np.array(rewards, dtype=np.float32)
    ev = rewards_arr.mean() if rewards_arr.size else 0.0
    std = rewards_arr.std(ddof=1) if rewards_arr.size > 1 else 0.0
    ci = 1.96 * std / np.sqrt(max(len(rewards_arr), 1))
    return EvalResult(
        ev_per_hand=float(ev),
        ev_per_100=float(ev * 100.0),
        ci_low=float(ev - ci),
        ci_high=float(ev + ci),
        win_rate=outcomes["win"] / max(hand_counter, 1),
        loss_rate=outcomes["loss"] / max(hand_counter, 1),
        push_rate=outcomes["push"] / max(hand_counter, 1),
    )
