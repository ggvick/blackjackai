"""Experience replay buffers for Rainbow DQN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplaySample:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_states: np.ndarray
    masks: np.ndarray
    next_masks: np.ndarray
    phases: np.ndarray
    next_phases: np.ndarray
    n_steps: np.ndarray
    indices: np.ndarray
    weights: np.ndarray


class PrioritizedReplayBuffer:
    """A simple Prioritized Experience Replay buffer.

    The implementation favours clarity over micro-optimisations.  Probabilities
    are recomputed on each sample which is adequate for training runs in the
    accompanying notebook and the automated tests.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        num_actions: int,
        *,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 1_000_000,
    ) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = max(beta_steps, 1)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.masks = np.ones((capacity, num_actions), dtype=np.float32)
        self.next_masks = np.ones((capacity, num_actions), dtype=np.float32)
        self.phases = np.zeros((capacity, 1), dtype=np.float32)
        self.next_phases = np.zeros((capacity, 1), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.n_steps = np.ones((capacity, 1), dtype=np.float32)

        self.position = 0
        self.size = 0
        self.max_priority = 1.0
        self.frame = 0

    def __len__(self) -> int:
        return self.size

    def beta(self) -> float:
        fraction = min(1.0, self.frame / float(self.beta_steps))
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_state: np.ndarray,
        mask: np.ndarray,
        next_mask: np.ndarray,
        phase: bool,
        next_phase: bool,
        steps: int,
    ) -> None:
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.next_states[idx] = next_state
        self.masks[idx] = mask
        self.next_masks[idx] = next_mask
        self.phases[idx] = float(phase)
        self.next_phases[idx] = float(next_phase)
        self.n_steps[idx] = float(steps)
        self.priorities[idx] = self.max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplaySample:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        self.frame += 1
        priorities = self.priorities[: self.size]
        scaled = priorities**self.alpha
        scaled_sum = scaled.sum()
        if scaled_sum <= 0:
            scaled = np.ones_like(priorities)
            scaled_sum = scaled.sum()
        probs = scaled / scaled_sum
        indices = np.random.choice(
            self.size, size=batch_size, replace=self.size < batch_size, p=probs
        )
        beta = self.beta()
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max() + 1e-6

        return ReplaySample(
            states=self.states[indices],
            actions=self.actions[indices].astype(np.int64),
            rewards=self.rewards[indices],
            dones=self.dones[indices],
            next_states=self.next_states[indices],
            masks=self.masks[indices],
            next_masks=self.next_masks[indices],
            phases=self.phases[indices],
            next_phases=self.next_phases[indices],
            n_steps=self.n_steps[indices],
            indices=indices,
            weights=weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float32).flatten()
        for idx, priority in zip(indices, priorities):
            value = float(max(priority, 1e-6))
            self.priorities[int(idx)] = value
            self.max_priority = max(self.max_priority, value)


__all__ = ["PrioritizedReplayBuffer", "ReplaySample"]
