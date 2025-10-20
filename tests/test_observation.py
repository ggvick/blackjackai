import numpy as np

from blackjackai_rl.env import BlackjackEnv, BlackjackEnvConfig


def test_observation_phase_and_ranges():
    config = BlackjackEnvConfig(reward_shaping=False, seed=99)
    env = BlackjackEnv(config)
    obs, info = env.reset()

    # Bet phase observation
    assert np.allclose(obs[:10], 0.0)
    assert obs[30] == 1.0  # phase bet indicator
    assert obs[31] == 0.0
    assert 0.0 <= obs[23] <= 1.0
    assert obs[25] == 0.0

    obs_play, _, _, info_play = env.step(0)
    assert obs_play.shape[0] == env.observation_builder.observation_dim
    assert np.isclose(obs_play[:10].sum(), 1.0)
    assert 0.0 <= obs_play[10] <= 1.0
    assert 0.0 <= obs_play[13] <= 1.0
    assert -1.0 <= obs_play[14] <= 1.0
    assert -1.0 <= obs_play[15] <= 1.0
    assert 0.0 <= obs_play[16] <= 1.0
    assert 0.0 <= obs_play[17] <= 1.0
    assert 0.0 <= obs_play[25] <= 1.0
    assert 0.0 <= obs_play[26] <= 1.0
    assert obs_play[30] == 0.0
    assert obs_play[31] == 1.0
