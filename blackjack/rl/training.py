"""Utilities for training and evaluating reinforcement learning agents."""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Optional, Tuple

import numpy as np

from .environment import BlackjackEnv


def seed_everything(seed: Optional[int] = None) -> None:
    """Seed Python and NumPy for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class TrainingConfig:
    """Configuration for Q-learning training."""

    episodes: int = 20000
    alpha: float = 0.05
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999
    log_every: int = 1000


@dataclass
class TrainingResult:
    """Aggregate metrics from a training run."""

    q_table: Dict[Tuple[int, int, int, int, int], np.ndarray]
    episode_stats: List[Dict[str, float]]
    summary: Dict[str, float]


class QLearningTrainer:
    """Simple tabular Q-learning trainer for the Blackjack environment."""

    def __init__(self, env: BlackjackEnv, config: Optional[TrainingConfig] = None, seed: Optional[int] = None) -> None:
        self.env = env
        self.config = config or TrainingConfig()
        self.seed = seed
        seed_everything(seed)

    def train(self) -> TrainingResult:
        q_table: MutableMapping[Tuple[int, int, int, int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.env.action_space), dtype=np.float32)
        )
        epsilon = self.config.epsilon_start
        outcomes = Counter()
        episode_stats: List[Dict[str, float]] = []
        total_reward = 0.0
        total_bet = 0.0
        start_time = time.perf_counter()

        for episode in range(1, self.config.episodes + 1):
            state = self.env.reset()
            info = self.env.last_info
            reward = self.env.last_reward
            episode_bet = info.get("bet", self.env.current_bet)

            # Handle automatic round completion (e.g. naturals)
            if self.env.round_complete:
                total_reward += reward
                total_bet += episode_bet
                outcomes[info.get("outcome", "push")] += 1
                self.env.round_complete = False
                epsilon = max(epsilon * self.config.epsilon_decay, self.config.epsilon_min)
                if episode % self.config.log_every == 0:
                    self._append_stats(episode_stats, outcomes, total_reward, total_bet, episode, epsilon)
                continue

            done = False
            episode_reward = 0.0

            while not done:
                if random.random() < epsilon:
                    action = self.env.sample_action()
                else:
                    action = int(np.argmax(q_table[state]))

                next_state, reward, done, info = self.env.step(action)
                best_next = 0.0 if done else float(np.max(q_table[next_state]))
                q_table[state][action] += self.config.alpha * (
                    reward + self.config.gamma * best_next - q_table[state][action]
                )
                state = next_state
                episode_reward += reward

            total_reward += episode_reward
            total_bet += episode_bet
            outcomes[info.get("outcome", "push")] += 1

            epsilon = max(epsilon * self.config.epsilon_decay, self.config.epsilon_min)

            if episode % self.config.log_every == 0:
                self._append_stats(episode_stats, outcomes, total_reward, total_bet, episode, epsilon)

        elapsed = time.perf_counter() - start_time
        wins = outcomes.get("win", 0)
        losses = outcomes.get("loss", 0)
        pushes = outcomes.get("push", 0)
        rounds = wins + losses + pushes
        summary = {
            "episodes": float(self.config.episodes),
            "rounds_played": float(rounds),
            "win_rate": wins / rounds if rounds else 0.0,
            "loss_rate": losses / rounds if rounds else 0.0,
            "push_rate": pushes / rounds if rounds else 0.0,
            "avg_reward": total_reward / rounds if rounds else 0.0,
            "avg_bet": total_bet / rounds if rounds else 0.0,
            "execution_time_sec": elapsed,
            "final_epsilon": epsilon,
        }

        frozen_table = {state: values.copy() for state, values in q_table.items()}
        return TrainingResult(q_table=frozen_table, episode_stats=episode_stats, summary=summary)

    @staticmethod
    def _append_stats(
        episode_stats: List[Dict[str, float]],
        outcomes: Counter,
        total_reward: float,
        total_bet: float,
        episode: int,
        epsilon: float,
    ) -> None:
        wins = outcomes.get("win", 0)
        losses = outcomes.get("loss", 0)
        pushes = outcomes.get("push", 0)
        rounds = wins + losses + pushes
        win_rate = wins / rounds if rounds else 0.0
        loss_rate = losses / rounds if rounds else 0.0
        push_rate = pushes / rounds if rounds else 0.0
        avg_reward = total_reward / rounds if rounds else 0.0
        avg_bet = total_bet / rounds if rounds else 0.0
        episode_stats.append(
            {
                "episode": float(episode),
                "epsilon": float(epsilon),
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "push_rate": push_rate,
                "avg_reward": avg_reward,
                "avg_bet": avg_bet,
            }
        )


def evaluate_policy(
    env: BlackjackEnv,
    q_table: Dict[Tuple[int, int, int, int, int], np.ndarray],
    episodes: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Run evaluation episodes using a greedy policy derived from the Q-table."""
    seed_everything(seed)
    outcomes = Counter()
    total_reward = 0.0
    total_bet = 0.0

    for _ in range(episodes):
        state = env.reset()
        info = env.last_info
        reward = env.last_reward
        episode_bet = info.get("bet", env.current_bet)

        if env.round_complete:
            total_reward += reward
            total_bet += episode_bet
            outcomes[info.get("outcome", "push")] += 1
            env.round_complete = False
            continue

        done = False
        episode_reward = 0.0

        while not done:
            action_values = q_table.get(state)
            if action_values is None:
                action = env.sample_action()
            else:
                action = int(np.argmax(action_values))
            state, reward, done, info = env.step(action)
            episode_reward += reward

        total_reward += episode_reward
        total_bet += episode_bet
        outcomes[info.get("outcome", "push")] += 1

    wins = outcomes.get("win", 0)
    losses = outcomes.get("loss", 0)
    pushes = outcomes.get("push", 0)
    rounds = wins + losses + pushes
    return {
        "episodes": float(episodes),
        "rounds_played": float(rounds),
        "win_rate": wins / rounds if rounds else 0.0,
        "loss_rate": losses / rounds if rounds else 0.0,
        "push_rate": pushes / rounds if rounds else 0.0,
        "avg_reward": total_reward / rounds if rounds else 0.0,
        "avg_bet": total_bet / rounds if rounds else 0.0,
    }
