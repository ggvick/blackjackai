"""Evaluation and reporting utilities for Blackjack agents."""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .agents import RainbowDQNAgent
from .env import BlackjackEnv, BlackjackEnvConfig
from .hand_logger import DecisionLogger, HandDecision
from .policies import (
    BasicStrategyPolicy,
    CountBettingPolicy,
    CountBettingSchedule,
    EvaluationPolicy,
)
from .strategy import basic_strategy
from .utils import ensure_dir


@dataclass
class EvaluationMetrics:
    ev_per_100: float
    ev_confidence: Tuple[float, float]
    win_rate: float
    push_rate: float
    loss_rate: float
    bust_rate: float
    total_hands: int
    bankroll_change: float
    action_frequencies: Dict[str, float]
    bet_by_true_count: Dict[str, float]
    outcome_by_true_count: Dict[str, float]
    baselines: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def _timestamp_dir(base: str | Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = ensure_dir(Path(base) / stamp)
    ensure_dir(path / "plots")
    ensure_dir(path / "logs")
    return path


def _compute_confidence_interval(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    margin = 1.96 * stdev / math.sqrt(len(values))
    return (mean - margin, mean + margin)


def _prepare_hand_record(
    env: BlackjackEnv,
    info: Dict[str, object],
    bet_index: int,
    bet_q_values: Sequence[float],
    bankroll_before: float,
) -> HandDecision:
    return HandDecision(
        episode_id=int(info.get("episode_id", env.episode_id)),
        shoe_id=int(info.get("shoe_id", env.shoe_id)),
        hand_id=int(info.get("hand_id", env.current_hand_index)),
        bet_index=bet_index,
        bet_amount=float(info.get("current_bet", env.last_bet)),
        bet_q_values=bet_q_values,
        bankroll_before=bankroll_before,
        running_count=float(env.counter.running_count),
        true_count=float(env.counter.true_count),
        penetration=float(env.penetration_progress),
        dealer_upcard=int(
            info.get(
                "dealer_upcard", env.dealer_cards[0].value if env.dealer_cards else 0
            )
        ),
        player_total=int(info.get("player_total", 0)),
    )


def _basic_strategy_action(env: BlackjackEnv, info: Dict[str, object]) -> str:
    hand = env.active_hands[env.current_hand_index]
    legal = env.valid_actions(hand)
    total, usable_ace = env.hand_total(hand.cards)
    is_pair, pair_rank = env.is_pair(hand.cards)
    dealer_upcard = env.dealer_cards[0].value if env.dealer_cards else 0
    decision = basic_strategy(
        total,
        usable_ace,
        pair_rank,
        dealer_upcard,
        env.ACTION_DOUBLE in legal,
        env.ACTION_SPLIT in legal,
        env.config.allow_surrender and len(hand.cards) == 2,
    )
    return decision.action


def _collect_deviation(
    action: int, env: BlackjackEnv, info: Dict[str, object]
) -> Optional[str]:
    recommended = _basic_strategy_action(env, info)
    chosen = env.action_names.get(action, "")
    if chosen != recommended:
        true_count = float(env.counter.true_count)
        return f"TC {true_count:.1f}: chose {chosen} over basic {recommended}"
    return None


def _bin_true_count(value: float) -> str:
    if value >= 5:
        return "5+"
    if value <= -5:
        return "-5-"
    return str(int(math.floor(value)))


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _generate_plots(
    records: Sequence[HandDecision],
    output_dir: Path,
    action_frequencies: Dict[str, float],
    bet_by_true_count: Dict[str, float],
    outcome_by_true_count: Dict[str, float],
) -> Dict[str, str]:
    plots: Dict[str, str] = {}
    indices = list(range(len(records)))
    bankroll = [record.bankroll_after for record in records]
    plt.figure(figsize=(10, 4))
    plt.plot(indices, bankroll, label="Bankroll")
    plt.xlabel("Hand index")
    plt.ylabel("Bankroll")
    plt.title("Bankroll trajectory")
    plt.grid(True)
    path = output_dir / "plots" / "bankroll_curve.png"
    _save_plot(path)
    plots["bankroll_curve"] = str(path)

    plt.figure(figsize=(8, 4))
    labels = list(action_frequencies.keys())
    values = [action_frequencies[label] for label in labels]
    plt.bar(labels, values, color="tab:blue")
    plt.ylabel("Frequency")
    plt.title("Action distribution")
    plt.grid(True, axis="y", alpha=0.3)
    path = output_dir / "plots" / "action_frequencies.png"
    _save_plot(path)
    plots["action_frequencies"] = str(path)

    if bet_by_true_count:
        plt.figure(figsize=(8, 4))
        labels = list(bet_by_true_count.keys())
        values = [bet_by_true_count[label] for label in labels]
        plt.bar(labels, values, color="tab:green")
        plt.ylabel("Average bet")
        plt.xlabel("True count bin")
        plt.title("Bet size vs. true count")
        plt.grid(True, axis="y", alpha=0.3)
        path = output_dir / "plots" / "bet_vs_true_count.png"
        _save_plot(path)
        plots["bet_vs_true_count"] = str(path)

    if outcome_by_true_count:
        plt.figure(figsize=(8, 4))
        labels = list(outcome_by_true_count.keys())
        values = [outcome_by_true_count[label] for label in labels]
        plt.bar(labels, values, color="tab:red")
        plt.ylabel("EV per hand")
        plt.xlabel("True count bin")
        plt.title("Outcome by true count")
        plt.grid(True, axis="y", alpha=0.3)
        path = output_dir / "plots" / "outcome_by_true_count.png"
        _save_plot(path)
        plots["outcome_by_true_count"] = str(path)

    return plots


def _summarise_llm(
    metrics: EvaluationMetrics,
    deviations: List[str],
    negative_q_patterns: List[str],
    training_history: Optional[Dict[str, Sequence[float]]] = None,
) -> str:
    lines = []
    lines.append("Blackjack RL Evaluation Summary")
    lines.append("=" * 36)
    lines.append(
        f"EV/100 hands: {metrics.ev_per_100:.3f} (95% CI {metrics.ev_confidence[0]:.3f} – {metrics.ev_confidence[1]:.3f})"
    )
    lines.append(
        f"Win/Loss/Push rates: {metrics.win_rate:.2%} / {metrics.loss_rate:.2%} / {metrics.push_rate:.2%}"
    )
    lines.append(f"Bust rate: {metrics.bust_rate:.2%}")
    if metrics.baselines:
        lines.append("Baseline EV/100:")
        for name, data in metrics.baselines.items():
            lines.append(f"  {name}: {data.get('ev_per_100', 0.0):.3f}")
    lines.append("Bet curve vs true count:")
    for bin_label, value in metrics.bet_by_true_count.items():
        lines.append(f"  TC {bin_label}: avg bet {value:.2f}")
    if deviations:
        lines.append("Strategy deviations at high counts:")
        for deviation in deviations[:5]:
            lines.append(f"  - {deviation}")
    if negative_q_patterns:
        lines.append("States with low Q-values:")
        for pattern in negative_q_patterns[:5]:
            lines.append(f"  - {pattern}")
    if training_history:
        losses = training_history.get("loss_play_history", [])
        if losses:
            lines.append(
                f"Training stability: play loss range {min(losses):.4f} – {max(losses):.4f}"
            )
    return "\n".join(lines)


def _compute_negative_q_patterns(records: Sequence[HandDecision]) -> List[str]:
    patterns: List[Tuple[float, str]] = []
    for record in records:
        for action in record.play_actions:
            q_values = action.get("q_values")
            if not q_values:
                continue
            max_q = max(q_values)
            description = f"Hand {record.hand_id} {action.get('action')} with obs hash {hash(tuple(action.get('observation', [])))}"
            patterns.append((max_q, description))
    patterns.sort(key=lambda item: item[0])
    return [desc for _val, desc in patterns[:5]]


def _evaluate_policy(
    policy: EvaluationPolicy, env_config: BlackjackEnvConfig, num_hands: int
) -> Dict[str, float]:
    env = BlackjackEnv(dataclasses.replace(env_config, reward_shaping=False))
    obs, info = env.reset()
    hands = 0
    total_profit = 0.0
    while hands < num_hands:
        action = policy.act(env, obs, info)
        obs, reward, done, info = env.step(action)
        total_profit += reward
        if info.get("hand_complete"):
            hands += 1
        if done:
            obs, info = env.reset()
    ev_per_100 = (total_profit / max(1, hands)) * 100.0
    return {"ev_per_100": ev_per_100}


def evaluate_agent(
    agent: RainbowDQNAgent,
    env_config: BlackjackEnvConfig,
    num_hands: int,
    output_dir: str | Path,
    *,
    training_history: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict[str, object]:
    output_path = _timestamp_dir(output_dir)
    env = BlackjackEnv(dataclasses.replace(env_config, reward_shaping=False))
    records: List[HandDecision] = []
    action_counter: Counter[str] = Counter()
    bet_bins: Dict[str, List[float]] = defaultdict(list)
    outcome_bins: Dict[str, List[float]] = defaultdict(list)
    deviations: List[str] = []

    logger = DecisionLogger(
        output_path / "logs" / "decisions.jsonl", output_path / "logs" / "decisions.csv"
    )

    obs, info = env.reset()
    active_records: Dict[int, HandDecision] = {}
    bankroll_after_last = env.bankroll
    hands_played = 0

    while hands_played < num_hands:
        if info.get("needs_bet", info.get("phase") == "bet"):
            q_bet, _ = agent.evaluate_q(np.expand_dims(obs, axis=0))
            action = agent.select_actions(
                np.expand_dims(obs, axis=0), [info], deterministic=True
            )[0]
            bankroll_before = env.bankroll
            obs, reward, done, info = env.step(action)
            info_for_hand = info
            hand_id = int(info_for_hand.get("hand_id", env.current_hand_index))
            record = _prepare_hand_record(
                env, info_for_hand, action, q_bet[0], bankroll_before
            )
            active_records[hand_id] = record
            bet_bins[_bin_true_count(record.true_count)].append(record.bet_amount)
            continue

        _, q_play = agent.evaluate_q(np.expand_dims(obs, axis=0))
        mask = np.asarray(info.get("action_mask"), dtype=np.float32)
        action = agent.select_actions(
            np.expand_dims(obs, axis=0), [info], deterministic=True
        )[0]
        obs_next, reward, done, info_next = env.step(action)
        hand_id = int(info.get("hand_id", env.current_hand_index))
        record = active_records.setdefault(
            hand_id, _prepare_hand_record(env, info, -1, [], env.bankroll)
        )
        record.play_actions.append(
            {
                "action": env.action_names.get(action, str(action)),
                "q_values": q_play[0].tolist(),
                "mask": mask.tolist(),
                "observation": obs.tolist(),
            }
        )
        action_counter[env.action_names.get(action, str(action))] += 1
        deviation = _collect_deviation(action, env, info)
        if deviation:
            deviations.append(deviation)
        if info_next.get("hand_complete"):
            record.outcome = str(info_next.get("outcome", ""))
            record.profit = float(info_next.get("delta", reward))
            record.bankroll_after = float(info_next.get("bankroll", env.bankroll))
            record.final_dealer_total = int(info_next.get("dealer_total", 0))
            record.final_player_total = int(info_next.get("player_total", 0))
            logger.log(record)
            records.append(record)
            outcome_bins[_bin_true_count(record.true_count)].append(record.profit)
            bankroll_after_last = record.bankroll_after
            active_records.pop(hand_id, None)
            hands_played += 1
        obs = obs_next
        info = info_next if not done else info_next.get("reset_info", info_next)
        if done:
            obs, info = env.reset()

    logger.close()

    profits = [record.profit for record in records]
    ev_per_hand = np.mean(profits) if profits else 0.0
    ev_per_100 = ev_per_hand * 100.0
    ci_low, ci_high = _compute_confidence_interval([p * 100.0 for p in profits])
    wins = sum(1 for p in profits if p > 0)
    pushes = sum(1 for p in profits if math.isclose(p, 0.0, abs_tol=1e-6))
    losses = len(profits) - wins - pushes
    total_hands = len(profits)
    win_rate = wins / total_hands if total_hands else 0.0
    push_rate = pushes / total_hands if total_hands else 0.0
    loss_rate = losses / total_hands if total_hands else 0.0
    bust_rate = 1.0 if bankroll_after_last <= env.config.bankroll_stop_loss else 0.0
    action_frequencies = {
        action: count / max(1, sum(action_counter.values()))
        for action, count in action_counter.items()
    }
    bet_by_true_count = {
        bin_label: float(np.mean(values)) for bin_label, values in bet_bins.items()
    }
    outcome_by_true_count = {
        bin_label: float(np.mean(values)) for bin_label, values in outcome_bins.items()
    }

    baselines: Dict[str, Dict[str, float]] = {}
    basic_policy = BasicStrategyPolicy(bet_index=0)
    count_policy = CountBettingPolicy(
        CountBettingSchedule(
            thresholds=[1, 2, 3], bet_indices=[1, 3, env_config.bet_actions - 1]
        )
    )
    baselines["basic_strategy"] = _evaluate_policy(
        basic_policy, env_config, min(num_hands, 5_000)
    )
    baselines["basic_with_count"] = _evaluate_policy(
        count_policy, env_config, min(num_hands, 5_000)
    )

    metrics = EvaluationMetrics(
        ev_per_100=float(ev_per_100),
        ev_confidence=(ci_low, ci_high),
        win_rate=float(win_rate),
        push_rate=float(push_rate),
        loss_rate=float(loss_rate),
        bust_rate=float(bust_rate),
        total_hands=total_hands,
        bankroll_change=float(bankroll_after_last - env.config.bankroll),
        action_frequencies=action_frequencies,
        bet_by_true_count=bet_by_true_count,
        outcome_by_true_count=outcome_by_true_count,
        baselines=baselines,
    )

    metrics_path = output_path / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, indent=2)

    plots = _generate_plots(
        records,
        output_path,
        action_frequencies,
        bet_by_true_count,
        outcome_by_true_count,
    )
    negative_patterns = _compute_negative_q_patterns(records)
    summary_text = _summarise_llm(
        metrics, deviations, negative_patterns, training_history
    )
    summary_path = output_path / "summary_llm.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    return {
        "metrics": metrics.to_dict(),
        "plots": plots,
        "records_path": str(output_path / "logs" / "decisions.jsonl"),
        "summary_path": str(summary_path),
        "run_dir": str(output_path),
    }


__all__ = ["evaluate_agent"]
