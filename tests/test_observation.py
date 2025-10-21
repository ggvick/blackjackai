import numpy as np

from blackjack_env.env import BlackjackEnv, EnvConfig


def set_cards(env, cards):
    env.state.shoe = list(reversed(cards))
    env.state.shoe_initial_count = len(env.state.shoe)
    env.step({"bet": 0})


def test_observation_values():
    env = BlackjackEnv(EnvConfig(num_decks=1, seed=21))
    env.reset()
    set_cards(env, [5, 10, 6, 9])
    obs = env._current_observation()
    space = env.observation_space
    assert obs.shape[0] == space.size
    assert np.isfinite(obs).all()

    dealer_one_hot = obs[:10]
    assert np.isclose(dealer_one_hot.sum(), 1.0)
    assert dealer_one_hot.argmax() == 9  # dealer upcard 10

    total_norm, is_soft, is_pair = obs[10:13]
    assert 0 <= total_norm <= 1
    assert is_soft == 0
    assert is_pair == 0

    true_count = obs[14]
    running_count = obs[15]
    decks_norm = obs[16]
    penetration = obs[17]
    assert -5 <= true_count <= 5
    assert -5 <= running_count <= 5
    assert 0 <= decks_norm <= 1
    assert 0 <= penetration <= 1

    last_action = obs[-5:]
    assert np.allclose(last_action, 0)
