import numpy as np

from blackjackai_rl.env import BlackjackEnv, BlackjackEnvConfig, Card


def make_card(rank: str, value: int) -> Card:
    return Card(rank=rank, value=value, suit="♠")


def rig_deck(env: BlackjackEnv, draw_sequence):
    env.shoe.cards = []
    filler = [make_card("2", 2)] * 16
    env.shoe.cards.extend(filler)
    for card in reversed(draw_sequence):
        env.shoe.cards.append(card)


def reset_with_deck(env: BlackjackEnv, draw_sequence):
    env.reset()
    rig_deck(env, draw_sequence)


def test_reward_win_loss_push():
    config = BlackjackEnvConfig(reward_shaping=False, allow_split=False, seed=123)
    env = BlackjackEnv(config)

    # Player 20 vs dealer 18 -> win
    reset_with_deck(
        env,
        [
            make_card("10", 10),
            make_card("8", 8),
            make_card("Queen", 10),
            make_card("King", 10),
        ],
    )
    env.step(0)  # bet
    _, reward, _, info = env.step(env.ACTION_STAND)
    assert np.isclose(reward, env.last_bet)
    assert info["outcome"] == "win"

    # Player 17 vs dealer 20 -> loss
    reset_with_deck(
        env,
        [
            make_card("10", 10),
            make_card("Queen", 10),
            make_card("9", 9),
            make_card("8", 8),
        ],
    )
    env.step(0)
    _, reward, _, info = env.step(env.ACTION_STAND)
    assert np.isclose(reward, -env.last_bet)
    assert info["outcome"] == "loss"

    # Push scenario 19 vs 19 -> zero reward
    reset_with_deck(
        env,
        [
            make_card("10", 10),
            make_card("9", 9),
            make_card("10", 10),
            make_card("9", 9),
        ],
    )
    env.step(0)
    _, reward, _, info = env.step(env.ACTION_STAND)
    assert np.isclose(reward, 0.0)
    assert info["outcome"] == "push"


def test_reward_natural_and_surrender():
    config = BlackjackEnvConfig(reward_shaping=False, allow_split=False, seed=77)
    env = BlackjackEnv(config)

    # Natural blackjack vs dealer 20 -> profit = 1.5 * bet
    reset_with_deck(
        env,
        [
            make_card("10", 10),
            make_card("Queen", 10),
            make_card("King", 10),
            make_card("Ace", 11),
        ],
    )
    _, reward, _, info = env.step(0)
    assert np.isclose(reward, env.last_bet * env.config.natural_payout)
    assert info["outcome"] in {"blackjack", "blackjack_win"}

    # Surrender: player 16 vs dealer 10
    reset_with_deck(
        env,
        [
            make_card("10", 10),
            make_card("Queen", 10),
            make_card("10", 10),
            make_card("6", 6),
        ],
    )
    env.step(0)
    _, reward, _, info = env.step(env.ACTION_SURRENDER)
    assert np.isclose(reward, -0.5 * env.last_bet)
    assert info["outcome"] == "surrender"


def test_reward_double_and_split():
    config = BlackjackEnvConfig(
        reward_shaping=False, allow_split=True, seed=5, max_splits=1
    )
    env = BlackjackEnv(config)

    # Double: player 11 vs dealer low card, draw 10 -> profit 2*bet
    reset_with_deck(
        env,
        [
            make_card("5", 5),
            make_card("9", 9),
            make_card("6", 6),
            make_card("5", 5),
            make_card("10", 10),
        ],
    )
    env.step(0)
    base_bet = env.last_bet
    _, reward, _, info = env.step(env.ACTION_DOUBLE)
    assert np.isclose(reward, 2 * base_bet)
    assert info["outcome"] == "win"

    # Split: pair of eights, both hands win
    reset_with_deck(
        env,
        [
            make_card("6", 6),
            make_card("10", 10),
            make_card("8", 8),
            make_card("8", 8),
            make_card("10", 10),
            make_card("9", 9),
            make_card("10", 10),
            make_card("9", 9),
        ],
    )
    env.step(0)
    # First action split
    env.step(env.ACTION_SPLIT)
    # First split hand: stand
    env.step(env.ACTION_STAND)
    # Second split hand: stand as well
    _, reward, _, info = env.step(env.ACTION_STAND)
    assert reward >= env.last_bet - 1e-6
    assert info["outcome"] in {"win", "stand"}
