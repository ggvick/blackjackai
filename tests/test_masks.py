from blackjack_env.env import BlackjackEnv, EnvConfig
from blackjack_env.masking import Action


def set_cards(env, cards):
    filler = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1] * 6
    env.state.shoe = list(reversed(cards + filler))
    env.state.shoe_initial_count = len(env.state.shoe)
    env.step({"bet": 0})


def test_action_masks_basic():
    env = BlackjackEnv(EnvConfig(seed=7))
    env.reset()
    # Pair for split
    set_cards(env, [8, 6, 8, 10])
    mask = env.available_actions()
    assert mask[Action.SPLIT]
    assert mask[Action.DOUBLE]
    assert mask[Action.SURRENDER]

    # After split first hand, double not allowed when more than two cards
    env.step(Action.SPLIT)
    env.step(Action.HIT)
    mask = env.available_actions()
    assert not mask[Action.DOUBLE]


def test_double_mask_after_action():
    env = BlackjackEnv(EnvConfig(seed=11))
    env.reset()
    set_cards(env, [6, 5, 5, 10])
    env.step(Action.DOUBLE)
    mask = env.available_actions()
    # After doubling the hand resolves and mask is zero as round complete
    assert mask.sum() == 0
