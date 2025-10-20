from blackjack_env.env import BlackjackEnv, EnvConfig
from blackjack_env.masking import Action


def run_round(env):
    env.step({"bet": 0})
    while env.state.stage == "play":
        env.step(Action.STAND)


def test_penetration_triggers_shuffle():
    config = EnvConfig(num_decks=1, penetration=0.1, seed=5)
    env = BlackjackEnv(config)
    env.reset()
    initial_length = len(env.state.shoe)
    reshuffled = False
    for _ in range(10):
        before = len(env.state.shoe)
        run_round(env)
        after = len(env.state.shoe)
        if after > before:
            reshuffled = True
            break
    assert reshuffled, "Expected shoe to reshuffle when penetration reached"
    assert env.state.shoe_initial_count == initial_length
