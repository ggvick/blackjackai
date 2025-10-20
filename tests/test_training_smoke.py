from blackjackai_rl.agents import DQNConfig
from blackjackai_rl.env import BlackjackEnvConfig
from blackjackai_rl.training import train_rainbow


def test_training_smoke(tmp_path):
    env_config = BlackjackEnvConfig(
        num_decks=1,
        penetration=0.5,
        reward_shaping=False,
        bankroll=50.0,
        bankroll_target=100.0,
        min_bet=1.0,
        max_bet=2.0,
        bet_actions=4,
        seed=7,
    )
    agent_config = DQNConfig(
        state_dim=36,
        num_actions=5,
        bet_actions=4,
        hidden_sizes=(64, 64),
        gamma=0.95,
        lr=1e-3,
        bet_lr=1e-3,
        epsilon_decay=500,
        batch_size=16,
        buffer_size=1_000,
        min_buffer_size=32,
        target_update_interval=100,
        prioritized_replay=True,
        atoms=31,
        v_min=-10.0,
        v_max=10.0,
        n_step=1,
        device="cpu",
    )
    results = train_rainbow(
        env_config,
        agent_config,
        total_steps=200,
        vector_envs=4,
        log_interval=100,
        output_dir=tmp_path,
    )
    assert results["reward_history"], "reward history should not be empty"
    assert results["loss_play_history"], "loss history should be tracked"
