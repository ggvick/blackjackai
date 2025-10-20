import numpy as np

from blackjackai_rl.env import BlackjackEnv, BlackjackEnvConfig, Phase


def test_reset_does_not_increment_steps():
    config = BlackjackEnvConfig(reward_shaping=False, seed=42)
    env = BlackjackEnv(config)
    obs, info = env.reset()

    assert env.step_count == 0
    assert env.episode_step == 0
    assert env.hands_played == 0
    assert env.locked_bet == 0.0
    assert env.phase is Phase.BET
    assert obs.shape[0] == env.observation_builder.observation_dim
    assert np.allclose(info["action_mask"], np.ones(env.num_actions))

    obs_after_bet, reward, done, info_after_bet = env.step(0)
    assert env.step_count == 1
    assert not done
    assert reward == 0.0
    assert obs_after_bet.shape[0] == env.observation_builder.observation_dim
    assert info_after_bet["phase"] == "play"
