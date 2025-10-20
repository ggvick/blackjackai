"""Utilities for logging Blackjack decision traces."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class HandDecision:
    episode_id: int
    shoe_id: int
    hand_id: int
    bet_index: int
    bet_amount: float
    bet_q_values: Sequence[float]
    play_actions: List[Dict[str, object]] = field(default_factory=list)
    outcome: str = ""
    profit: float = 0.0
    bankroll_before: float = 0.0
    bankroll_after: float = 0.0
    running_count: float = 0.0
    true_count: float = 0.0
    penetration: float = 0.0
    dealer_upcard: int = 0
    player_total: int = 0
    final_dealer_total: int = 0
    final_player_total: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "shoe_id": self.shoe_id,
            "hand_id": self.hand_id,
            "bet_index": self.bet_index,
            "bet_amount": self.bet_amount,
            "bet_q_values": list(self.bet_q_values),
            "play_actions": self.play_actions,
            "outcome": self.outcome,
            "profit": self.profit,
            "bankroll_before": self.bankroll_before,
            "bankroll_after": self.bankroll_after,
            "running_count": self.running_count,
            "true_count": self.true_count,
            "penetration": self.penetration,
            "dealer_upcard": self.dealer_upcard,
            "player_total": self.player_total,
            "final_dealer_total": self.final_dealer_total,
            "final_player_total": self.final_player_total,
        }


class DecisionLogger:
    """Write decision traces to JSONL and CSV simultaneously."""

    def __init__(self, jsonl_path: str | Path, csv_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_file = self.jsonl_path.open("w", encoding="utf-8")
        self.csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        fieldnames = [
            "episode_id",
            "shoe_id",
            "hand_id",
            "bet_index",
            "bet_amount",
            "outcome",
            "profit",
            "bankroll_before",
            "bankroll_after",
            "running_count",
            "true_count",
            "penetration",
            "dealer_upcard",
            "player_total",
            "final_dealer_total",
            "final_player_total",
            "bet_q_values",
            "play_actions",
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, decision: HandDecision) -> None:
        record = decision.to_dict()
        json.dump(record, self.json_file)
        self.json_file.write("\n")
        csv_record = {**record}
        csv_record["bet_q_values"] = json.dumps(record["bet_q_values"])
        csv_record["play_actions"] = json.dumps(record["play_actions"])
        self.writer.writerow(csv_record)

    def close(self) -> None:
        self.json_file.close()
        self.csv_file.close()

    def __enter__(self) -> "DecisionLogger":  # pragma: no cover - convenience wrapper
        return self

    def __exit__(
        self, exc_type, exc, tb
    ) -> None:  # pragma: no cover - convenience wrapper
        self.close()


__all__ = ["HandDecision", "DecisionLogger"]
