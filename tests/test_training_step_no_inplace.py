import numpy as np
import pytest
import torch

from agents.dqn_rainbow import AgentConfig, RainbowDQNAgent


def _populate_buffer(agent: RainbowDQNAgent, transitions: int = 4) -> None:
    obs_dim = agent.config.observation_dim
    play_actions = agent.config.play_actions
    for _ in range(transitions):
        state = np.random.randn(obs_dim).astype(np.float32)
        next_state = np.random.randn(obs_dim).astype(np.float32)
        mask = np.ones(play_actions, dtype=bool)
        next_mask = np.ones(play_actions, dtype=bool)
        bet_action = np.random.randint(agent.config.bet_actions)
        action = np.random.randint(play_actions)
        reward = float(np.random.randn())
        done = False
        agent.store(state, mask, bet_action, action, reward, next_state, next_mask, done)


def _run_training_step(device: str, use_amp: bool, enable_c51: bool) -> None:
    config = AgentConfig(
        observation_dim=8,
        bet_actions=3,
        play_actions=4,
        num_atoms=11,
        buffer_size=128,
        min_buffer_size=2,
        batch_size=2,
        n_step=1,
        device=device,
        use_amp=use_amp,
        enable_c51=enable_c51,
        use_noisy=False,
        replay_on_gpu=False,
        compile_model=False,
    )
    agent = RainbowDQNAgent(config)
    _populate_buffer(agent, transitions=8)
    metrics = agent.train_step()
    assert metrics["loss"] >= 0.0


def test_training_step_cpu_backward() -> None:
    _run_training_step(device="cpu", use_amp=False, enable_c51=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is required for AMP test")
def test_training_step_amp_backward() -> None:
    _run_training_step(device="cuda", use_amp=True, enable_c51=True)
