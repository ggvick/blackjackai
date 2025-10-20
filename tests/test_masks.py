from blackjackai_rl.env import BlackjackEnv, BlackjackEnvConfig, Card


def make_card(rank: str, value: int) -> Card:
    return Card(rank=rank, value=value, suit="♣")


def rig_deck(env: BlackjackEnv, draw_sequence):
    env.shoe.cards = [make_card("2", 2) for _ in range(16)]
    for card in reversed(draw_sequence):
        env.shoe.cards.append(card)


def reset_with_deck(env: BlackjackEnv, draw_sequence):
    env.reset()
    rig_deck(env, draw_sequence)


def test_action_mask_allows_split_and_double():
    config = BlackjackEnvConfig(reward_shaping=False, seed=11)
    env = BlackjackEnv(config)
    reset_with_deck(
        env,
        [
            make_card("6", 6),
            make_card("5", 5),
            make_card("8", 8),
            make_card("8", 8),
        ],
    )
    obs, reward, done, info = env.step(0)
    assert reward == 0.0
    assert not done
    mask = info["action_mask"]
    assert mask.shape == (env.num_actions,)
    assert mask[env.ACTION_SPLIT] == 1.0
    assert mask[env.ACTION_DOUBLE] == 1.0
    assert mask[env.ACTION_STAND] == 1.0

    # After hit, double and split should be disabled
    obs, reward, done, info = env.step(env.ACTION_HIT)
    assert not done
    mask_after_hit = info["action_mask"]
    assert mask_after_hit[env.ACTION_DOUBLE] == 0.0
    assert mask_after_hit[env.ACTION_SPLIT] == 0.0


def test_mask_blocks_double_when_bankroll_low():
    config = BlackjackEnvConfig(
        reward_shaping=False, bankroll=1.0, min_bet=1.0, max_bet=8.0, seed=21
    )
    env = BlackjackEnv(config)
    reset_with_deck(
        env,
        [
            make_card("9", 9),
            make_card("7", 7),
            make_card("10", 10),
            make_card("9", 9),
        ],
    )
    _, reward, done, info = env.step(0)
    assert reward == 0.0
    assert not done
    mask = info["action_mask"]
    assert mask[env.ACTION_DOUBLE] == 0.0
