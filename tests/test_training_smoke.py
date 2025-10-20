import numpy as np
from blackjack_env.env import BlackjackEnv, EnvConfig
from agents.dqn_rainbow import AgentConfig, RainbowDQNAgent


def test_training_smoke_runs():
    env = BlackjackEnv(EnvConfig(seed=123, min_bet=1.0, max_bet=4.0, bet_levels=4))
    env.reset()
    config = AgentConfig(
        observation_dim=env.observation_space.size,
        bet_actions=env.config.bet_levels,
        buffer_size=2048,
        min_buffer_size=64,
        batch_size=64,
        target_update_interval=250,
        enable_c51=False,
        use_amp=False,
        device="cpu",
        epsilon_decay=1000,
    )
    agent = RainbowDQNAgent(config)
    metrics = agent.train(env, steps=300)
    assert "loss" in metrics
    assert np.isfinite(metrics["loss"])
    assert agent.global_step >= 300
