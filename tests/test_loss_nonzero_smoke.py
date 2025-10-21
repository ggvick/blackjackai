import math
from typing import List

import numpy as np
import torch

from agents.dqn_rainbow import AgentConfig, RainbowDQNAgent


def _make_agent() -> RainbowDQNAgent:
    return RainbowDQNAgent(
        AgentConfig(
            observation_dim=8,
            bet_actions=3,
            play_actions=4,
            hidden_sizes=(64, 64),
            num_atoms=11,
            buffer_size=4096,
            min_buffer_size=64,
            batch_size=64,
            n_step=1,
            use_amp=torch.cuda.is_available(),
            replay_on_gpu=torch.cuda.is_available(),
            compile_model=False,
            prioritized_replay=True,
            epsilon_decay=1000,
        )
    )


def _random_transition(agent: RainbowDQNAgent):
    obs_dim = agent.config.observation_dim
    play_actions = agent.config.play_actions
    state = np.random.randn(obs_dim).astype(np.float32)
    next_state = np.random.randn(obs_dim).astype(np.float32)
    mask = np.random.rand(play_actions) > 0.2
    if not mask.any():
        mask[0] = True
    next_mask = np.random.rand(play_actions) > 0.2
    if not next_mask.any():
        next_mask[0] = True
    bet_action = np.random.randint(agent.config.bet_actions)
    action = np.random.randint(play_actions)
    reward = float(np.random.randn())
    done = bool(np.random.rand() < 0.05)
    return state, mask, bet_action, action, reward, next_state, next_mask, done


def test_loss_remains_positive_after_warmup():
    np.random.seed(0)
    torch.manual_seed(0)
    agent = _make_agent()
    for _ in range(agent.config.min_buffer_size):
        agent.store(*_random_transition(agent))
    losses: List[float] = []
    updates = 128 if torch.cuda.is_available() else 64
    for _ in range(updates):
        metrics = agent.train_step()
        if metrics["loss"] > 0:
            losses.append(metrics["loss"])
    mean_loss = float(np.mean(losses)) if losses else 0.0
    assert mean_loss > 0.0
    assert math.isfinite(mean_loss)
