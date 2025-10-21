import numpy as np

from blackjack_env.env import BlackjackEnv, EnvConfig


def test_reset_stage_and_counters():
    env = BlackjackEnv(EnvConfig(seed=42))
    obs = env.reset()
    assert env.state.round_number == 0
    assert env.state.stage == "bet"
    assert np.allclose(obs, 0)

    obs, reward, done, info = env.step({"bet": 0})
    assert env.state.stage == "play"
    assert env.state.round_number == 1
    assert reward == 0.0

    env.reset()
    assert env.state.round_number == 0
    assert env.state.stage == "bet"
