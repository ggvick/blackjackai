"""Evaluation and visualization helpers for Blackjack RL."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .agents import BasicStrategyAgent, DQNAgent
from .env import BlackjackEnv, BlackjackEnvConfig
from .training import legal_mask_from_info
# No direct filesystem helpers required here; plotting functions ensure directories.


@dataclass
class HandRecord:
    index: int
    bankroll_before: float
    bankroll_after: float
    bet: float
    delta: float
    reward: float
    outcome: str
    true_count: float
    running_count: float
    action: str
    decks_remaining: float


@dataclass
class EvaluationSummary:
    win_rate: float
    expected_value: float
    total_hands: int
    bankroll_change: float
    episodes_played: int
    average_bet: float


def evaluate_policy(
    env_config: BlackjackEnvConfig,
    action_fn: Callable[[np.ndarray, Sequence[int], Dict[str, float | str], BlackjackEnv], int],
    num_hands: int = 2_000,
    seed: int | None = None,
) -> Dict[str, object]:
    """Run evaluation episodes and collect rich telemetry."""

    config = dataclasses.replace(env_config)
    config.reward_shaping = False
    config.seed = seed
    env = BlackjackEnv(config)
    obs, info = env.reset()
    hand_records: List[HandRecord] = []
    bankroll_before = env.bankroll
    hands_played = 0
    episode_returns: List[float] = []
    session_return = 0.0

    while hands_played < num_hands:
        legal_actions = info.get("legal_actions") or env.valid_actions(env.active_hands[env.current_hand_index])
        mask = legal_actions if isinstance(legal_actions, Sequence) else env.valid_actions(env.active_hands[env.current_hand_index])
        action = action_fn(obs, mask, info, env)
        obs, reward, done, info = env.step(action)
        session_return += reward
        if info.get("hand_complete"):
            hand_records.append(
                HandRecord(
                    index=hands_played,
                    bankroll_before=bankroll_before,
                    bankroll_after=float(info.get("bankroll", env.bankroll)),
                    bet=float(info.get("bet", env.config.min_bet)),
                    delta=float(info.get("delta", reward)),
                    reward=float(reward),
                    outcome=str(info.get("outcome", "")),
                    true_count=float(info.get("true_count", 0.0)),
                    running_count=float(info.get("running_count", 0.0)),
                    action=env.action_names.get(action, ""),
                    decks_remaining=float(env.counter.decks_remaining),
                )
            )
            bankroll_before = env.bankroll
            hands_played += 1
        if done:
            episode_returns.append(session_return)
            session_return = 0.0
            if hands_played >= num_hands:
                break
            obs, info = env.reset()
            bankroll_before = env.bankroll
            continue

    bankroll_change = env.bankroll - env.config.bankroll
    wins = sum(1 for record in hand_records if record.delta > 0)
    pushes = sum(1 for record in hand_records if np.isclose(record.delta, 0.0))
    losses = len(hand_records) - wins - pushes
    win_rate = wins / max(1, len(hand_records))
    expected_value = np.mean([record.delta for record in hand_records]) if hand_records else 0.0
    average_bet = np.mean([record.bet for record in hand_records]) if hand_records else 0.0

    summary = EvaluationSummary(
        win_rate=win_rate,
        expected_value=expected_value,
        total_hands=len(hand_records),
        bankroll_change=bankroll_change,
        episodes_played=len(episode_returns),
        average_bet=average_bet,
    )

    return {
        "hand_records": hand_records,
        "episode_returns": episode_returns,
        "summary": summary,
    }


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], output_dir: str | Path) -> Dict[str, str]:
    output_paths: Dict[str, str] = {}
    rewards = history.get("reward_history", [])
    losses = history.get("loss_history", [])
    q_values = history.get("q_history", [])
    epsilons = history.get("epsilon_history", [])
    steps = np.arange(len(rewards))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, rewards, label="Average reward")
    plt.xlabel("Batch step")
    plt.ylabel("Reward")
    plt.title("DQN reward curve")
    plt.grid(True)
    plot_path = Path(output_dir) / "plots" / "training_reward_curve.png"
    _save_plot(plot_path)
    output_paths["reward_curve"] = str(plot_path)

    if losses:
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(losses)), losses, label="Loss", color="tab:red")
        plt.xlabel("Update step")
        plt.ylabel("Loss")
        plt.title("DQN loss curve")
        plt.grid(True)
        plot_path = Path(output_dir) / "plots" / "training_loss_curve.png"
        _save_plot(plot_path)
        output_paths["loss_curve"] = str(plot_path)

    if q_values:
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(q_values)), q_values, label="Q-value", color="tab:green")
        plt.xlabel("Update step")
        plt.ylabel("Average Q")
        plt.title("Estimated Q-values")
        plt.grid(True)
        plot_path = Path(output_dir) / "plots" / "training_q_curve.png"
        _save_plot(plot_path)
        output_paths["q_curve"] = str(plot_path)

    if epsilons:
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(epsilons)), epsilons, label="Epsilon", color="tab:orange")
        plt.xlabel("Batch step")
        plt.ylabel("Epsilon")
        plt.title("Exploration schedule")
        plt.grid(True)
        plot_path = Path(output_dir) / "plots" / "epsilon_curve.png"
        _save_plot(plot_path)
        output_paths["epsilon_curve"] = str(plot_path)

    return output_paths


def plot_evaluation_results(records: Iterable[HandRecord], output_dir: str | Path) -> Dict[str, str]:
    records = list(records)
    output_paths: Dict[str, str] = {}
    if not records:
        return output_paths

    indices = [r.index for r in records]
    bankroll = [r.bankroll_after for r in records]
    deltas = [r.delta for r in records]
    bets = [r.bet for r in records]
    true_counts = [r.true_count for r in records]
    actions = [r.action for r in records]
    decks_remaining = [r.decks_remaining for r in records]

    plt.figure(figsize=(8, 4))
    plt.plot(indices, bankroll, label="Bankroll")
    plt.xlabel("Hand index")
    plt.ylabel("Bankroll ($)")
    plt.title("Bankroll trajectory")
    plt.grid(True)
    path = Path(output_dir) / "plots" / "bankroll_curve.png"
    _save_plot(path)
    output_paths["bankroll_curve"] = str(path)

    plt.figure(figsize=(8, 4))
    plt.hist(deltas, bins=30, color="tab:blue", alpha=0.7)
    plt.xlabel("Hand return ($)")
    plt.ylabel("Frequency")
    plt.title("Distribution of hand returns")
    path = Path(output_dir) / "plots" / "hand_return_hist.png"
    _save_plot(path)
    output_paths["hand_return_hist"] = str(path)

    plt.figure(figsize=(8, 4))
    count_buckets = np.round(true_counts).astype(int)
    unique_buckets = sorted(set(count_buckets))
    action_map: Dict[str, List[int]] = {action: [] for action in set(actions)}
    for bucket, action in zip(count_buckets, actions):
        action_map.setdefault(action, []).append(bucket)
    for action, bucket_values in action_map.items():
        counts = [bucket_values.count(bucket) for bucket in unique_buckets]
        plt.plot(unique_buckets, counts, label=action)
    plt.xlabel("True count bucket")
    plt.ylabel("Action usage")
    plt.title("Action usage by true count")
    plt.legend()
    plt.grid(True)
    path = Path(output_dir) / "plots" / "action_vs_true_count.png"
    _save_plot(path)
    output_paths["action_vs_true_count"] = str(path)

    plt.figure(figsize=(8, 4))
    plt.scatter(true_counts, bets, alpha=0.6)
    plt.xlabel("True count")
    plt.ylabel("Bet size ($)")
    plt.title("Bet size vs true count")
    plt.grid(True)
    path = Path(output_dir) / "plots" / "bet_vs_true_count.png"
    _save_plot(path)
    output_paths["bet_vs_true_count"] = str(path)

    plt.figure(figsize=(8, 4))
    plt.scatter(decks_remaining, bets, alpha=0.6, c=np.arange(len(bets)), cmap="viridis")
    plt.xlabel("Decks remaining")
    plt.ylabel("Bet size ($)")
    plt.title("Shoe penetration vs bet aggressiveness")
    plt.grid(True)
    path = Path(output_dir) / "plots" / "penetration_vs_bet.png"
    _save_plot(path)
    output_paths["penetration_vs_bet"] = str(path)

    return output_paths


def save_hand_records(records: Iterable[HandRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "index",
        "bankroll_before",
        "bankroll_after",
        "bet",
        "delta",
        "reward",
        "outcome",
        "true_count",
        "running_count",
        "action",
        "decks_remaining",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for record in records:
            handle.write(",".join(str(getattr(record, field)) for field in header) + "\n")


def compare_to_baseline(
    env_config: BlackjackEnvConfig,
    trained_agent: DQNAgent,
    num_hands: int = 2_000,
) -> Dict[str, object]:
    """Evaluate trained agent versus the deterministic baseline."""

    def dqn_action(obs: np.ndarray, legal_actions: Sequence[int], info: Dict[str, float | str], env: BlackjackEnv) -> int:
        mask = legal_mask_from_info({"legal_actions": legal_actions}, trained_agent.config.num_actions)
        actions = trained_agent.greedy_actions(obs[None, :], mask[None, :])
        return int(actions[0])

    def basic_action(_: np.ndarray, __: Sequence[int], ___: Dict[str, float | str], env: BlackjackEnv) -> int:
        baseline = BasicStrategyAgent()
        return baseline.select_action(env)

    trained_results = evaluate_policy(env_config, dqn_action, num_hands=num_hands)
    baseline_results = evaluate_policy(env_config, basic_action, num_hands=num_hands)

    gain = trained_results["summary"].expected_value - baseline_results["summary"].expected_value

    return {
        "trained": trained_results,
        "baseline": baseline_results,
        "expected_value_gain": gain,
    }


__all__ = [
    "HandRecord",
    "EvaluationSummary",
    "evaluate_policy",
    "plot_training_curves",
    "plot_evaluation_results",
    "save_hand_records",
    "compare_to_baseline",
]
