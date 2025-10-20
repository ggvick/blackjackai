"""Training utilities for Blackjack RL agents."""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .agents import DQNAgent, DQNConfig
from .env import BlackjackEnvConfig, VectorizedBlackjackEnv
from .utils import TimingInfo, ensure_dir


@dataclass
class CurriculumStage:
    threshold: int
    allow_double: bool
    allow_split: bool


def legal_mask_from_info(info: Dict[str, float | Sequence[int]], num_actions: int) -> np.ndarray:
    mask = np.zeros(num_actions, dtype=np.float32)
    legal = info.get("legal_actions")
    if isinstance(legal, Sequence):
        for action in legal:
            mask[int(action)] = 1.0
    if mask.sum() == 0:
        mask[:] = 1.0
    return mask


def apply_curriculum(envs: VectorizedBlackjackEnv, stage: CurriculumStage) -> None:
    for env in envs.envs:
        env.config.allow_double = stage.allow_double
        env.config.allow_split = stage.allow_split


def default_curriculum(total_steps: int) -> List[CurriculumStage]:
    return [
        CurriculumStage(int(total_steps * 0.2), False, False),
        CurriculumStage(int(total_steps * 0.6), True, False),
        CurriculumStage(total_steps + 1, True, True),
    ]


def train_dqn(
    env_config: BlackjackEnvConfig,
    agent_config: DQNConfig,
    total_steps: int,
    vector_envs: int = 32,
    log_interval: int = 1_000,
    curriculum: Iterable[CurriculumStage] | None = None,
    output_dir: str | Path = "outputs",
) -> Dict[str, object]:
    """Train a DQN agent and persist checkpoints/metrics."""

    ensure_dir(output_dir)
    models_dir = ensure_dir(Path(output_dir) / "models")
    env_conf_copy = dataclasses.replace(env_config)
    vec_env = VectorizedBlackjackEnv(vector_envs, env_conf_copy)
    observations, info_list = vec_env.reset()
    agent = DQNAgent(agent_config)

    legal_masks = np.stack(
        [legal_mask_from_info(info, agent_config.num_actions) for info in info_list]
    )

    reward_history: List[float] = []
    loss_history: List[float] = []
    q_history: List[float] = []
    epsilon_history: List[float] = []
    timing_records: List[TimingInfo] = []

    stages = list(curriculum or default_curriculum(total_steps))
    current_stage_index = 0
    apply_curriculum(vec_env, stages[current_stage_index])

    best_moving_avg = -np.inf
    best_path = models_dir / "best_dqn.pt"
    total_env_steps = 0
    recent_rewards: List[float] = []

    start_time = time.perf_counter()
    while total_env_steps < total_steps:
        actions = agent.select_actions(observations, legal_masks)
        next_obs, rewards, dones, infos = vec_env.step(actions)
        next_masks = np.stack(
            [legal_mask_from_info(info, agent_config.num_actions) for info in infos]
        )

        for idx in range(vector_envs):
            agent.add_transition(
                observations[idx],
                int(actions[idx]),
                float(rewards[idx]),
                bool(dones[idx]),
                next_obs[idx],
                legal_masks[idx],
                next_masks[idx],
            )

        update = agent.train_step()
        if update:
            loss_history.append(update.get("loss", 0.0))
            q_history.append(update.get("q_mean", 0.0))

        batch_reward = float(np.mean(rewards))
        reward_history.append(batch_reward)
        epsilon_history.append(agent.epsilon())
        recent_rewards.append(batch_reward)
        if len(recent_rewards) > 1_000:
            recent_rewards.pop(0)

        total_env_steps += vector_envs

        if total_env_steps >= stages[current_stage_index].threshold and current_stage_index + 1 < len(stages):
            current_stage_index += 1
            apply_curriculum(vec_env, stages[current_stage_index])

        if total_env_steps % log_interval == 0:
            moving_avg = float(np.mean(recent_rewards[-log_interval:])) if recent_rewards else 0.0
            elapsed = time.perf_counter() - start_time
            timing_records.append(TimingInfo(label=f"step_{total_env_steps}", duration_seconds=elapsed))
            if moving_avg > best_moving_avg:
                agent.save(str(best_path))
                best_moving_avg = moving_avg

        observations = next_obs
        legal_masks = next_masks

    if best_moving_avg == -np.inf:
        agent.save(str(best_path))

    total_duration = time.perf_counter() - start_time
    timing_records.append(TimingInfo(label="total_training", duration_seconds=total_duration))

    agent.save(str(models_dir / "final_dqn.pt"))

    return {
        "reward_history": reward_history,
        "loss_history": loss_history,
        "q_history": q_history,
        "epsilon_history": epsilon_history,
        "best_model_path": str(best_path),
        "timings": timing_records,
    }


__all__ = ["train_dqn", "CurriculumStage", "default_curriculum", "legal_mask_from_info"]
