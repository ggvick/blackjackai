"""Prioritized experience replay buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def _powerlaw(alpha: float, values: np.ndarray) -> np.ndarray:
    return np.power(values + 1e-6, alpha)


@dataclass
class PrioritizedReplayBuffer:
    capacity: int
    alpha: float = 0.6
    beta: float = 0.4
    beta_increment: float = 1e-6

    def __post_init__(self) -> None:
        self.pos = 0
        self.full = False
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.storage: List[Tuple] = [None] * self.capacity

    def add(self, transition: Tuple) -> None:
        max_priority = self.priorities.max() if self.pos > 0 or self.full else 1.0
        self.storage[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        if self.full:
            probabilities = _powerlaw(self.alpha, self.priorities)
        else:
            probabilities = _powerlaw(self.alpha, self.priorities[: self.pos])
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(probabilities), batch_size, p=probabilities)
        samples = [self.storage[idx] for idx in indices]
        weights = (len(probabilities) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)
