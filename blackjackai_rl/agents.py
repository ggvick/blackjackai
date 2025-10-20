"""Agent implementations used in the Blackjack RL notebook."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
except ImportError:  # pragma: no cover - torch available in notebook
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Adam = None  # type: ignore

from .env import BlackjackEnv
from .strategy import basic_strategy


# ---------------------------------------------------------------------------
# Baseline basic-strategy agent
# ---------------------------------------------------------------------------


class BasicStrategyAgent:
    """Deterministic agent using the standard basic strategy tables."""

    def select_action(self, env: BlackjackEnv) -> int:
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
        action = env._strategy_to_action(decision.action)
        if action in legal:
            return action
        if env.ACTION_HIT in legal:
            return env.ACTION_HIT
        return env.ACTION_STAND


# ---------------------------------------------------------------------------
# Tabular Q-learning agent
# ---------------------------------------------------------------------------


@dataclass
class TabularConfig:
    gamma: float = 0.99
    lr: float = 0.1
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: int = 50_000


class TabularQLearningAgent:
    """Simple Q-table baseline for sanity checking."""

    def __init__(self, num_actions: int, config: TabularConfig) -> None:
        self.num_actions = num_actions
        self.config = config
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.steps = 0

    @staticmethod
    def _discretize(obs: np.ndarray) -> Tuple[int, ...]:
        player_total = int(round(obs[0] * 21))
        is_soft = int(obs[1] > 0.5)
        is_pair = int(obs[2] > 0.5)
        dealer_upcard = int(round(obs[3] * 11))
        true_count_bucket = int(np.clip(round(obs[4] * 10), -6, 6))
        bet_bucket = int(round(obs[6] * 10))
        split_count = int(round(obs[11]))
        return (
            player_total,
            is_soft,
            is_pair,
            dealer_upcard,
            true_count_bucket,
            bet_bucket,
            split_count,
        )

    def _epsilon(self) -> float:
        frac = min(1.0, self.steps / float(self.config.epsilon_decay))
        return self.config.epsilon_start + frac * (self.config.epsilon_final - self.config.epsilon_start)

    def select_action(self, obs: np.ndarray, legal_actions: Sequence[int]) -> int:
        self.steps += 1
        epsilon = self._epsilon()
        state = self._discretize(obs)
        if random.random() < epsilon:
            return int(random.choice(list(legal_actions)))

        q_values = self.q_table.setdefault(state, np.zeros(self.num_actions, dtype=np.float32))
        legal_values = [(q_values[a], a) for a in legal_actions]
        if not legal_values:
            return 0
        return int(max(legal_values)[1])

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        legal_next: Sequence[int],
    ) -> None:
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)
        q_values = self.q_table.setdefault(state, np.zeros(self.num_actions, dtype=np.float32))
        next_values = self.q_table.setdefault(next_state, np.zeros(self.num_actions, dtype=np.float32))
        best_next = max((next_values[a] for a in legal_next), default=0.0)
        target = reward + (0.0 if done else self.config.gamma * best_next)
        q_values[action] += self.config.lr * (target - q_values[action])


# ---------------------------------------------------------------------------
# Deep Q-Network agent
# ---------------------------------------------------------------------------


@dataclass
class DQNConfig:
    state_dim: int
    num_actions: int
    hidden_sizes: Tuple[int, int] = (256, 256)
    gamma: float = 0.99
    lr: float = 5e-4
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: int = 200_000
    batch_size: int = 256
    buffer_size: int = 200_000
    min_buffer_size: int = 2_000
    target_update_interval: int = 2_000
    tau: float = 0.005
    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, num_actions: int) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, num_actions), dtype=np.float32)
        self.next_legal_masks = np.zeros((capacity, num_actions), dtype=np.float32)
        self.index = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_state: np.ndarray,
        legal_mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = float(done)
        self.next_states[self.index] = next_state
        self.legal_masks[self.index] = legal_mask
        self.next_legal_masks[self.index] = next_mask
        self.index = (self.index + 1) % self.capacity
        self.full = self.full or self.index == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.index

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(len(self), size=batch_size, replace=False)
        return {
            "states": self.states[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "dones": self.dones[idxs],
            "next_states": self.next_states[idxs],
            "legal_masks": self.legal_masks[idxs],
            "next_legal_masks": self.next_legal_masks[idxs],
        }


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_sizes: Tuple[int, int]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layers(x)


class DQNAgent:
    def __init__(self, config: DQNConfig) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for DQNAgent")
        self.config = config
        self.device = torch.device(config.device)
        self.q_network = QNetwork(config.state_dim, config.num_actions, config.hidden_sizes).to(self.device)
        self.target_network = QNetwork(config.state_dim, config.num_actions, config.hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = Adam(self.q_network.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size, config.state_dim, config.num_actions)
        self.steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.steps / float(self.config.epsilon_decay))
        return self.config.epsilon_start + frac * (self.config.epsilon_final - self.config.epsilon_start)

    def select_actions(self, obs_batch: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
        batch_size = obs_batch.shape[0]
        actions = np.zeros(batch_size, dtype=np.int64)
        eps = self.epsilon()
        self.steps += batch_size
        if torch is None:
            raise ImportError("PyTorch is required for DQNAgent")

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_batch).to(self.device)
            q_values = self.q_network(obs_tensor).cpu().numpy()
        for i in range(batch_size):
            mask = legal_masks[i]
            legal_indices = np.flatnonzero(mask)
            if legal_indices.size == 0:
                actions[i] = 0
                continue
            if random.random() < eps:
                actions[i] = int(random.choice(list(legal_indices)))
            else:
                masked = np.where(mask > 0, q_values[i], -1e9)
                actions[i] = int(np.argmax(masked))
        return actions

    def add_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_state: np.ndarray,
        legal_mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        self.buffer.add(state, action, reward, done, next_state, legal_mask, next_mask)

    def train_step(self) -> Dict[str, float]:
        if len(self.buffer) < self.config.min_buffer_size:
            return {}
        batch_size = min(self.config.batch_size, len(self.buffer))
        batch = self.buffer.sample(batch_size)
        states = torch.from_numpy(batch["states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        legal_masks = torch.from_numpy(batch["legal_masks"]).to(self.device)
        next_legal_masks = torch.from_numpy(batch["next_legal_masks"]).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values[next_legal_masks <= 0] = -1e9
            next_max = torch.max(next_q_values, dim=1, keepdim=True).values
            target = rewards + (1 - dones) * self.config.gamma * next_max
        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()

        if self.steps % self.config.target_update_interval == 0:
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
                )

        return {"loss": float(loss.item()), "q_mean": float(q_values.mean().item())}

    def greedy_actions(self, obs_batch: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
        if torch is None:
            raise ImportError("PyTorch is required for DQNAgent")
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_batch).to(self.device)
            q_values = self.q_network(obs_tensor).cpu().numpy()
        actions = np.zeros(obs_batch.shape[0], dtype=np.int64)
        for i in range(obs_batch.shape[0]):
            mask = legal_masks[i]
            legal_indices = np.flatnonzero(mask)
            if legal_indices.size == 0:
                actions[i] = 0
            else:
                masked = np.where(mask > 0, q_values[i], -1e9)
                actions[i] = int(np.argmax(masked))
        return actions

    def save(self, path: str) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for DQNAgent")
        payload = {
            "model_state": self.q_network.state_dict(),
            "target_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "steps": self.steps,
            "config": self.config.__dict__,
        }
        torch.save(payload, path)

    def load(self, path: str, map_location: str | None = None) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for DQNAgent")
        payload = torch.load(path, map_location=map_location)
        self.q_network.load_state_dict(payload["model_state"])
        self.target_network.load_state_dict(payload["target_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.steps = int(payload.get("steps", 0))


__all__ = [
    "BasicStrategyAgent",
    "TabularConfig",
    "TabularQLearningAgent",
    "DQNConfig",
    "DQNAgent",
    "ReplayBuffer",
]
