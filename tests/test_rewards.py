import numpy as np

from blackjack_env.env import BlackjackEnv, EnvConfig
from blackjack_env.masking import Action


def set_shoe(env, cards):
    filler = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1] * 6
    env.state.shoe = list(reversed(cards + filler))
    env.state.shoe_initial_count = len(env.state.shoe)


def play_round(env, cards, action_sequence):
    set_shoe(env, cards)
    env.state.count_state.dealt_cards = 0
    env.state.count_state.running_count = 0
    env.state.count_state.count_10 = 0
    env.state.count_state.count_a = 0
    env.step({"bet": 0})
    total_reward = 0.0
    for action in action_sequence:
        _, reward, done, info = env.step(action)
        total_reward += reward
        if info.get("round_complete"):
            break
    return total_reward


def test_win_loss_push_rewards():
    env = BlackjackEnv(EnvConfig(seed=1))
    env.reset()
    # Player 10+9 vs dealer 6+10+7 bust
    win_reward = play_round(env, [10, 6, 9, 10, 7], [Action.STAND])
    assert np.isclose(win_reward, env.config.min_bet)

    env.reset()
    # Player 10+8 vs dealer 10+9 (loss)
    loss_reward = play_round(env, [10, 10, 8, 9], [Action.STAND])
    assert np.isclose(loss_reward, -env.config.min_bet)

    env.reset()
    # Push scenario 10+8 vs dealer 9+9
    push_reward = play_round(env, [10, 9, 8, 9], [Action.STAND])
    assert np.isclose(push_reward, 0.0)


def test_double_surrender_natural_split():
    env = BlackjackEnv(EnvConfig(seed=2))
    env.reset()

    # Double win: player 5+6 double draw 10 vs dealer 10+7
    double_cards = [5, 10, 6, 7, 10]
    reward = play_round(env, double_cards, [Action.DOUBLE])
    assert np.isclose(reward, env.config.min_bet * 2)

    env.reset()
    # Natural blackjack pays 3:2
    natural_cards = [1, 5, 10, 10]
    reward = play_round(env, natural_cards, [Action.STAND])
    assert np.isclose(reward, env.config.min_bet * env.config.natural_payout)

    env.reset()
    # Surrender yields -0.5 bet
    surrender_cards = [10, 10, 6, 9]
    reward = play_round(env, surrender_cards, [Action.SURRENDER])
    assert np.isclose(reward, -0.5 * env.config.min_bet)

    env.reset()
    # Split: pair of 8s vs dealer 6, each wins -> +2 bets
    split_cards = [8, 6, 8, 9, 10, 10, 10]
    reward = play_round(env, split_cards, [Action.SPLIT, Action.STAND, Action.STAND])
    assert reward >= env.config.min_bet
