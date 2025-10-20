from blackjackai_rl.env import BlackjackEnv, BlackjackEnvConfig


def test_penetration_triggers_shuffle():
    config = BlackjackEnvConfig(
        num_decks=1, penetration=0.25, reward_shaping=False, seed=13
    )
    env = BlackjackEnv(config)
    env.reset()
    initial_shoe = env.shoe_id
    cards_to_draw = int(env.total_cards * config.penetration) + 1
    for _ in range(cards_to_draw):
        env.draw_card()
    # Shuffle is triggered lazily when the next round is about to start
    obs, reward, done, info = env.step(0)
    assert not done
    assert reward == 0.0
    assert env.shoe_id == initial_shoe + 1
    # After reshuffle and initial deal we should only have the cards from the new hand drawn
    assert env.cards_drawn == 4
