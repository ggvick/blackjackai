"""Prioritized experience replay buffer with optional GPU storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


def _powerlaw(alpha: float, values: np.ndarray) -> np.ndarray:
    if alpha <= 0.0:
        return np.ones_like(values, dtype=np.float32)
    return np.power(values + 1e-6, alpha, dtype=np.float64).astype(np.float32)


@dataclass
class PrioritizedReplayBuffer:
    capacity: int
    observation_dim: int
    action_dim: int
    device: torch.device
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_increment: float = 1e-6
    use_amp: bool = False
    replay_on_gpu: bool = False

    def __post_init__(self) -> None:
        self.pos = 0
        self.full = False
        self.beta = float(self.beta_start)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        if self.replay_on_gpu and self.device.type == "cuda":
            obs_dtype = torch.float16 if self.use_amp else torch.float32
            self.states = torch.empty(
                (self.capacity, self.observation_dim),
                dtype=obs_dtype,
                device=self.device,
            )
            self.next_states = torch.empty_like(self.states)
            self.masks = torch.empty(
                (self.capacity, self.action_dim), dtype=torch.bool, device=self.device
            )
            self.next_masks = torch.empty_like(self.masks)
            self.bet_actions = torch.empty(
                (self.capacity,), dtype=torch.long, device=self.device
            )
            self.actions = torch.empty_like(self.bet_actions)
            reward_dtype = torch.float16 if self.use_amp else torch.float32
            self.rewards = torch.empty(
                (self.capacity,), dtype=reward_dtype, device=self.device
            )
            self.dones = torch.empty((self.capacity,), dtype=torch.bool, device=self.device)
            self.storage: List[Tuple] = []  # kept for typing; unused in GPU path
        else:
            self.replay_on_gpu = False
            self.storage = [None] * self.capacity  # type: ignore[assignment]

    # ------------------------------------------------------------------
    def add(self, transition: Tuple) -> None:
        max_priority = self.priorities.max() if (self.full or self.pos > 0) else 1.0
        if self.replay_on_gpu:
            (
                state,
                mask,
                bet_action,
                action,
                reward,
                next_state,
                next_mask,
                done,
            ) = transition
            idx = self.pos
            state_tensor = torch.as_tensor(state, device=self.device, dtype=self.states.dtype)
            next_state_tensor = torch.as_tensor(
                next_state, device=self.device, dtype=self.next_states.dtype
            )
            mask_tensor = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
            next_mask_tensor = torch.as_tensor(next_mask, device=self.device, dtype=torch.bool)
            reward_tensor = torch.as_tensor(reward, device=self.device, dtype=self.rewards.dtype)
            self.states[idx].copy_(state_tensor)
            self.next_states[idx].copy_(next_state_tensor)
            self.masks[idx].copy_(mask_tensor)
            self.next_masks[idx].copy_(next_mask_tensor)
            self.bet_actions[idx] = int(bet_action)
            self.actions[idx] = int(action)
            self.rewards[idx] = reward_tensor
            self.dones[idx] = bool(done)
        else:
            self.storage[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor | np.ndarray], np.ndarray, np.ndarray | None]:
        valid_length = self.capacity if self.full else self.pos
        if valid_length == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        probs = _powerlaw(self.alpha, self.priorities[:valid_length])
        probs /= probs.sum()
        indices = np.random.choice(valid_length, batch_size, p=probs)
        if self.alpha <= 0.0:
            weights = None
        else:
            weights = (valid_length * probs[indices]) ** (-self.beta)
            weights /= weights.max()
        self.beta = min(self.beta_end, self.beta + self.beta_increment)
        if self.replay_on_gpu:
            batch = {
                "states": self.states[indices],
                "next_states": self.next_states[indices],
                "legal_mask": self.masks[indices],
                "legal_mask_next": self.next_masks[indices],
                "bet_actions": self.bet_actions[indices],
                "actions": self.actions[indices],
                "rewards": self.rewards[indices],
                "dones": self.dones[indices],
            }
        else:
            transitions = [self.storage[idx] for idx in indices]
            (
                states,
                masks,
                bet_actions,
                actions,
                rewards,
                next_states,
                next_masks,
                dones,
            ) = zip(*transitions)
            batch = {
                "states": np.asarray(states, dtype=np.float32),
                "next_states": np.asarray(next_states, dtype=np.float32),
                "legal_mask": np.asarray(masks, dtype=bool),
                "legal_mask_next": np.asarray(next_masks, dtype=bool),
                "bet_actions": np.asarray(bet_actions, dtype=np.int64),
                "actions": np.asarray(actions, dtype=np.int64),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "dones": np.asarray(dones, dtype=bool),
            }
        if weights is not None:
            weights = weights.astype(np.float32)
        return batch, indices, weights

    # ------------------------------------------------------------------
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)
