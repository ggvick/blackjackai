import math
from typing import Dict

import torch

from agents.dqn_rainbow import AgentConfig, RainbowDQNAgent


def _make_config() -> AgentConfig:
    return AgentConfig(
        observation_dim=8,
        bet_actions=3,
        play_actions=4,
        hidden_sizes=(64, 64),
        num_atoms=11,
        buffer_size=256,
        min_buffer_size=8,
        batch_size=8,
        n_step=1,
        use_amp=torch.cuda.is_available(),
        replay_on_gpu=torch.cuda.is_available(),
        compile_model=False,
        prioritized_replay=True,
        epsilon_decay=1000,
    )


def make_fake_batch_on_device(device: torch.device) -> Dict[str, torch.Tensor]:
    torch.manual_seed(0)
    batch_size = 8
    obs_dim = 8
    play_actions = 4
    states = torch.randn(batch_size, obs_dim, device=device)
    next_states = torch.randn(batch_size, obs_dim, device=device)
    legal = torch.ones(batch_size, play_actions, dtype=torch.bool, device=device)
    bet_actions = torch.randint(0, 3, (batch_size,), device=device)
    actions = torch.randint(0, play_actions, (batch_size,), device=device)
    rewards = torch.randn(batch_size, device=device)
    dones = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return {
        "states": states,
        "next_states": next_states,
        "legal_mask": legal,
        "legal_mask_next": legal.clone(),
        "bet_actions": bet_actions,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def test_models_on_cuda_if_available():
    agent = RainbowDQNAgent(_make_config())
    if torch.cuda.is_available():
        assert agent.device.type == "cuda"


def test_train_step_forward_backward():
    agent = RainbowDQNAgent(_make_config())
    batch = make_fake_batch_on_device(agent.device)
    loss = agent._debug_train_step_once(batch)
    value = float(loss)
    assert value >= 0.0 and math.isfinite(value)
