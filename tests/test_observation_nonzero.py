import numpy as np

from blackjack_env.env import BlackjackEnv, EnvConfig


def test_observation_contains_signal():
    env = BlackjackEnv(EnvConfig(seed=7))
    obs = env.reset()
    assert obs.shape[0] == env.observation_space.size
    obs, _, _, _ = env.step({"bet": 0})
    x = np.asarray(obs, dtype=np.float32)
    assert x.shape[0] == env.observation_space.size
    assert np.isfinite(x).all()
    assert (x != 0).any()
    dealer_one_hot = x[:10]
    assert np.isclose(dealer_one_hot.sum(), 1.0)
