"""Training utilities for the Rainbow Blackjack agent."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from tqdm.auto import tqdm

from .agents import DQNConfig, RainbowDQNAgent
from .env import BlackjackEnvConfig, VectorizedBlackjackEnv
from .utils import TimingInfo, ensure_dir


@dataclass
class TrainingStats:
    reward_history: List[float]
    loss_play_history: List[float]
    loss_bet_history: List[float]
    epsilon_history: List[float]


def _info_for_next(info: Dict[str, object]) -> Dict[str, object]:
    if "reset_info" in info:
        return info["reset_info"]  # type: ignore[index]
    return info


def train_rainbow(
    env_config: BlackjackEnvConfig,
    agent_config: DQNConfig,
    total_steps: int,
    *,
    vector_envs: int = 32,
    log_interval: int = 2_000,
    evaluation_hands: int | None = None,
    output_dir: str | Path = "runs",
) -> Dict[str, object]:
    """Train the RainbowDQN agent and persist checkpoints/metrics."""

    ensure_dir(output_dir)
    models_dir = ensure_dir(Path(output_dir) / "models")
    env_conf_copy = dataclasses.replace(env_config)
    vec_env = VectorizedBlackjackEnv(vector_envs, env_conf_copy)
    observations, info_list = vec_env.reset()

    agent = RainbowDQNAgent(agent_config)
    stats = TrainingStats([], [], [], [])
    timing: List[TimingInfo] = []
    best_reward = -np.inf
    best_path = models_dir / "best_rainbow.pt"

    total_env_steps = 0
    recent_rewards: List[float] = []
    start_time = time.perf_counter()

    progress = tqdm(
        total=total_steps, desc="Training", unit="frame", disable=total_steps <= 0
    )
    while total_env_steps < total_steps:
        actions = agent.select_actions(observations, info_list)
        next_obs, rewards, dones, next_infos = vec_env.step(actions)

        for idx in range(vector_envs):
            next_info = _info_for_next(next_infos[idx])
            agent.add_transition(
                env_index=idx,
                state=observations[idx],
                action=int(actions[idx]),
                reward=float(rewards[idx]),
                done=bool(dones[idx]),
                next_state=next_obs[idx],
                info=info_list[idx],
                next_info=next_info,
            )
            next_infos[idx] = next_info

        update = agent.train_step()
        if update is not None:
            stats.loss_play_history.append(update.get("loss_play", 0.0))
            stats.loss_bet_history.append(update.get("loss_bet", 0.0))
            stats.epsilon_history.append(update.get("epsilon", 0.0))
        else:
            stats.loss_play_history.append(0.0)
            stats.loss_bet_history.append(0.0)
            stats.epsilon_history.append(agent.epsilon())

        batch_reward = float(np.mean(rewards))
        stats.reward_history.append(batch_reward)
        recent_rewards.append(batch_reward)
        if len(recent_rewards) > log_interval:
            recent_rewards.pop(0)

        total_env_steps += vector_envs
        progress.update(vector_envs)
        observations = next_obs
        info_list = [next_infos[idx] for idx in range(vector_envs)]

        if total_env_steps % log_interval == 0:
            moving_avg = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            elapsed = time.perf_counter() - start_time
            timing.append(
                TimingInfo(label=f"step_{total_env_steps}", duration_seconds=elapsed)
            )
            if moving_avg > best_reward:
                agent.save(str(best_path))
                best_reward = moving_avg

    if best_reward == -np.inf:
        agent.save(str(best_path))

    total_duration = time.perf_counter() - start_time
    timing.append(TimingInfo(label="total_training", duration_seconds=total_duration))
    agent.save(str(models_dir / "final_rainbow.pt"))
    progress.close()

    return {
        "reward_history": stats.reward_history,
        "loss_play_history": stats.loss_play_history,
        "loss_bet_history": stats.loss_bet_history,
        "epsilon_history": stats.epsilon_history,
        "best_model_path": str(best_path),
        "timings": timing,
    }


__all__ = ["train_rainbow"]
