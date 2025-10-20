"""Agent implementations used in the Blackjack RL project."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from itertools import chain
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim

from .env import BlackjackEnv
from .masking import apply_action_mask
from .networks import RainbowDQN
from .replay import PrioritizedReplayBuffer
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

    def select_bet(self, env: BlackjackEnv) -> int:
        return 0  # always choose the minimum bet option


# ---------------------------------------------------------------------------
# Tabular Q-learning agent (for sanity checks / tests)
# ---------------------------------------------------------------------------


@dataclass
class TabularConfig:
    gamma: float = 0.99
    lr: float = 0.1
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: int = 50_000


class TabularQLearningAgent:
    """Simple Q-table baseline for smoke tests."""

    def __init__(self, num_actions: int, config: TabularConfig) -> None:
        self.num_actions = num_actions
        self.config = config
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.steps = 0

    @staticmethod
    def _discretize(obs: np.ndarray) -> Tuple[int, ...]:
        # crude discretisation tailored for the richer observation vector
        dealer_bucket = int(np.argmax(obs[:10]))
        player_total = int(round(obs[10] * 21))
        is_soft = int(obs[11] > 0.5)
        split_count = int(round(obs[13] * 10))
        true_count_bucket = int(np.clip(round(obs[15] * 10), -10, 10))
        running_count_bucket = int(np.clip(round(obs[14] * 20), -20, 20))
        phase = int(obs[30] < 0.5)  # play phase indicator
        return (
            dealer_bucket,
            player_total,
            is_soft,
            split_count,
            true_count_bucket,
            running_count_bucket,
            phase,
        )

    def _epsilon(self) -> float:
        frac = min(1.0, self.steps / float(self.config.epsilon_decay))
        return self.config.epsilon_start + frac * (
            self.config.epsilon_final - self.config.epsilon_start
        )

    def select_action(self, obs: np.ndarray, legal_actions: Sequence[int]) -> int:
        self.steps += 1
        epsilon = self._epsilon()
        state = self._discretize(obs)
        if random.random() < epsilon:
            return int(random.choice(list(legal_actions)))

        q_values = self.q_table.setdefault(
            state, np.zeros(self.num_actions, dtype=np.float32)
        )
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
        q_values = self.q_table.setdefault(
            state, np.zeros(self.num_actions, dtype=np.float32)
        )
        next_values = self.q_table.setdefault(
            next_state, np.zeros(self.num_actions, dtype=np.float32)
        )
        best_next = max((next_values[a] for a in legal_next), default=0.0)
        target = reward + (0.0 if done else self.config.gamma * best_next)
        q_values[action] += self.config.lr * (target - q_values[action])


# ---------------------------------------------------------------------------
# Rainbow DQN agent with betting & playing heads
# ---------------------------------------------------------------------------


@dataclass
class DQNConfig:
    state_dim: int
    num_actions: int
    bet_actions: int
    hidden_sizes: Tuple[int, int] = (512, 512)
    gamma: float = 0.99
    lr: float = 3e-4
    bet_lr: float = 3e-4
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: int = 1_200_000
    batch_size: int = 512
    buffer_size: int = 800_000
    min_buffer_size: int = 20_000
    grad_clip: float = 5.0
    double_dqn: bool = True
    dueling: bool = True
    prioritized_replay: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 1_200_000
    n_step: int = 3
    distributional_c51: bool = True
    atoms: int = 51
    v_min: float = -20.0
    v_max: float = 20.0
    noisy_nets: bool = False
    target_update_interval: int | None = 15_000
    tau: float | None = None
    device: str = "cpu"


@dataclass
class NStepTransition:
    state: np.ndarray
    action: int
    reward: float
    mask: np.ndarray
    phase: bool
    done: bool
    next_state: np.ndarray
    next_mask: np.ndarray
    next_phase: bool


class RainbowDQNAgent:
    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.policy_net = RainbowDQN(
            state_dim=config.state_dim,
            hidden_sizes=config.hidden_sizes,
            bet_actions=config.bet_actions,
            play_actions=config.num_actions,
            atoms=config.atoms,
            noisy=config.noisy_nets,
            dueling=config.dueling,
        ).to(self.device)
        self.target_net = RainbowDQN(
            state_dim=config.state_dim,
            hidden_sizes=config.hidden_sizes,
            bet_actions=config.bet_actions,
            play_actions=config.num_actions,
            atoms=config.atoms,
            noisy=config.noisy_nets,
            dueling=config.dueling,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer_play = optim.AdamW(
            chain(
                self.policy_net.feature_layers.parameters(),
                self.policy_net.play_head.parameters(),
            ),
            lr=config.lr,
        )
        self.optimizer_bet = optim.AdamW(
            self.policy_net.bet_head.parameters(), lr=config.bet_lr
        )

        buffer_kwargs = {}
        if config.prioritized_replay:
            buffer_kwargs = dict(
                alpha=config.per_alpha,
                beta_start=config.per_beta_start,
                beta_end=config.per_beta_end,
                beta_steps=config.per_beta_steps,
            )
        self.replay = PrioritizedReplayBuffer(
            capacity=config.buffer_size,
            state_dim=config.state_dim,
            num_actions=config.num_actions,
            **buffer_kwargs,
        )

        self.n_step_buffers: Dict[int, deque[NStepTransition]] = defaultdict(deque)
        self.total_frames = 0
        self.training_steps = 0
        support = torch.linspace(
            config.v_min, config.v_max, config.atoms, device=self.device
        )
        self.support = support
        self.delta_z = float((config.v_max - config.v_min) / (config.atoms - 1))

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        if self.config.noisy_nets:
            return 0.0
        frac = min(1.0, self.total_frames / float(max(1, self.config.epsilon_decay)))
        return self.config.epsilon_start + frac * (
            self.config.epsilon_final - self.config.epsilon_start
        )

    def select_actions(
        self,
        observations: np.ndarray,
        infos: Sequence[Dict[str, object]],
        deterministic: bool = False,
    ) -> np.ndarray:
        obs_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        if self.config.noisy_nets:
            self.policy_net.reset_noise()
        with torch.no_grad():
            bet_atoms, play_atoms = self.policy_net(obs_tensor)
            bet_probs = torch.softmax(bet_atoms, dim=-1)
            play_probs = torch.softmax(play_atoms, dim=-1)
            bet_q = torch.sum(bet_probs * self.support, dim=-1)
            play_q = torch.sum(play_probs * self.support, dim=-1)
        epsilon = 0.0 if (deterministic or self.config.noisy_nets) else self.epsilon()
        actions: List[int] = []
        for idx, info in enumerate(infos):
            needs_bet = bool(info.get("needs_bet", info.get("phase") == "bet"))
            if needs_bet:
                num_bet_actions = self.config.bet_actions
                q_values = bet_q[idx].detach().cpu().numpy()
                if not deterministic and random.random() < epsilon:
                    action = random.randrange(num_bet_actions)
                else:
                    action = int(np.argmax(q_values))
            else:
                mask = np.asarray(info.get("action_mask"), dtype=np.float32)
                if mask.shape[0] != self.config.num_actions:
                    mask = np.ones(self.config.num_actions, dtype=np.float32)
                q_values = play_q[idx].detach().cpu().numpy()
                masked = apply_action_mask(q_values, mask)
                legal = np.where(mask > 0.5)[0]
                if len(legal) == 0:
                    action = 0
                elif not deterministic and random.random() < epsilon:
                    action = int(random.choice(list(legal)))
                else:
                    action = int(np.argmax(masked))
            actions.append(action)
        return np.asarray(actions, dtype=np.int64)

    def evaluate_q(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            bet_atoms, play_atoms = self.policy_net(obs_tensor)
            bet_probs = torch.softmax(bet_atoms, dim=-1)
            play_probs = torch.softmax(play_atoms, dim=-1)
            bet_q = torch.sum(bet_probs * self.support, dim=-1)
            play_q = torch.sum(play_probs * self.support, dim=-1)
        return bet_q.cpu().numpy(), play_q.cpu().numpy()

    # ------------------------------------------------------------------
    # Experience storage with n-step returns
    # ------------------------------------------------------------------
    def add_transition(
        self,
        env_index: int,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_state: np.ndarray,
        info: Dict[str, object],
        next_info: Dict[str, object],
    ) -> None:
        mask = self._mask_from_info(info)
        next_mask = self._mask_from_info(next_info)
        phase = bool(info.get("needs_bet", info.get("phase") == "bet"))
        next_phase = bool(next_info.get("needs_bet", next_info.get("phase") == "bet"))
        transition = NStepTransition(
            state=np.asarray(state, dtype=np.float32).copy(),
            action=int(action),
            reward=float(reward),
            mask=mask,
            phase=phase,
            done=bool(done),
            next_state=np.asarray(next_state, dtype=np.float32).copy(),
            next_mask=next_mask,
            next_phase=next_phase,
        )
        buffer = self.n_step_buffers[env_index]
        buffer.append(transition)
        self.total_frames += 1
        self._maybe_store_transition(env_index)
        if done:
            while self._maybe_store_transition(env_index):
                pass
            buffer.clear()

    def _mask_from_info(self, info: Dict[str, object]) -> np.ndarray:
        needs_bet = bool(info.get("needs_bet", info.get("phase") == "bet"))
        if needs_bet:
            return np.ones(self.config.num_actions, dtype=np.float32)
        mask = info.get("action_mask")
        if mask is None:
            return np.ones(self.config.num_actions, dtype=np.float32)
        mask_arr = np.asarray(mask, dtype=np.float32)
        if mask_arr.shape[0] != self.config.num_actions:
            mask_arr = np.ones(self.config.num_actions, dtype=np.float32)
        return mask_arr.copy()

    def _maybe_store_transition(self, env_index: int) -> bool:
        buffer = self.n_step_buffers[env_index]
        if not buffer:
            return False
        total_reward = 0.0
        discount = 1.0
        steps = 0
        final_transition: NStepTransition | None = None
        for transition in buffer:
            total_reward += discount * transition.reward
            steps += 1
            final_transition = transition
            if transition.done or steps >= self.config.n_step:
                break
            discount *= self.config.gamma
        if final_transition is None:
            return False
        if not final_transition.done and steps < self.config.n_step:
            return False
        first_transition = buffer[0]
        self.replay.add(
            state=first_transition.state,
            action=first_transition.action,
            reward=total_reward,
            done=final_transition.done,
            next_state=final_transition.next_state,
            mask=first_transition.mask,
            next_mask=final_transition.next_mask,
            phase=first_transition.phase,
            next_phase=final_transition.next_phase,
            steps=steps,
        )
        buffer.popleft()
        return True

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_step(self) -> Dict[str, float] | None:
        if len(self.replay) < self.config.min_buffer_size:
            return None
        batch_size = min(self.config.batch_size, len(self.replay))
        sample = self.replay.sample(batch_size)
        states = torch.as_tensor(sample.states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            sample.next_states, dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            sample.actions.squeeze(-1), dtype=torch.int64, device=self.device
        )
        rewards = torch.as_tensor(
            sample.rewards.squeeze(-1), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            sample.dones.squeeze(-1), dtype=torch.float32, device=self.device
        )
        next_masks = torch.as_tensor(
            sample.next_masks, dtype=torch.float32, device=self.device
        )
        phases = torch.as_tensor(
            sample.phases.squeeze(-1), dtype=torch.bool, device=self.device
        )
        next_phases = torch.as_tensor(
            sample.next_phases.squeeze(-1), dtype=torch.bool, device=self.device
        )
        n_steps = torch.as_tensor(
            sample.n_steps.squeeze(-1), dtype=torch.float32, device=self.device
        )
        weights = torch.as_tensor(
            sample.weights, dtype=torch.float32, device=self.device
        )

        self.policy_net.train()
        if self.config.noisy_nets:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

        bet_atoms, play_atoms = self.policy_net(states)
        bet_log_probs = torch.log_softmax(bet_atoms, dim=-1)
        play_log_probs = torch.log_softmax(play_atoms, dim=-1)

        with torch.no_grad():
            target_bet_atoms, target_play_atoms = self.target_net(next_states)
            target_bet_probs = torch.softmax(target_bet_atoms, dim=-1)
            target_play_probs = torch.softmax(target_play_atoms, dim=-1)
            if self.config.double_dqn:
                policy_next_bet_atoms, policy_next_play_atoms = self.policy_net(
                    next_states
                )
                policy_next_bet_probs = torch.softmax(policy_next_bet_atoms, dim=-1)
                policy_next_play_probs = torch.softmax(policy_next_play_atoms, dim=-1)
            else:
                policy_next_bet_probs = target_bet_probs
                policy_next_play_probs = target_play_probs

            policy_next_bet_q = torch.sum(policy_next_bet_probs * self.support, dim=-1)
            policy_next_play_q = torch.sum(
                policy_next_play_probs * self.support, dim=-1
            )
            policy_next_play_q = policy_next_play_q.masked_fill(next_masks < 0.5, -1e9)
            next_bet_actions = policy_next_bet_q.argmax(dim=-1)
            next_play_actions = policy_next_play_q.argmax(dim=-1)

            next_dists = torch.zeros(batch_size, self.config.atoms, device=self.device)
            if next_phases.any():
                bet_indices = torch.nonzero(next_phases, as_tuple=False).squeeze(-1)
                next_dists[bet_indices] = target_bet_probs[
                    bet_indices, next_bet_actions[bet_indices], :
                ]
            play_indices = torch.nonzero(~next_phases, as_tuple=False).squeeze(-1)
            if play_indices.numel() > 0:
                next_dists[play_indices] = target_play_probs[
                    play_indices, next_play_actions[play_indices], :
                ]

            projected = self._project_distribution(next_dists, rewards, dones, n_steps)

        priorities = torch.zeros(batch_size, device=self.device)
        loss_play_value = torch.tensor(0.0, device=self.device)
        loss_bet_value = torch.tensor(0.0, device=self.device)

        bet_loss: torch.Tensor | None = None
        play_loss: torch.Tensor | None = None

        bet_indices = torch.nonzero(phases, as_tuple=False).squeeze(-1)
        if bet_indices.numel() > 0:
            bet_logits = bet_log_probs[bet_indices]
            bet_actions = actions[bet_indices]
            action_selector = bet_actions.view(-1, 1, 1).expand(
                -1, 1, self.config.atoms
            )
            chosen = bet_logits.gather(1, action_selector).squeeze(1)
            target = projected[bet_indices]
            per_sample = -(target * chosen).sum(dim=-1)
            bet_loss = (per_sample * weights[bet_indices]).mean()
            priorities[bet_indices] = per_sample.detach()
            loss_bet_value = bet_loss.detach()

        play_indices = torch.nonzero(~phases, as_tuple=False).squeeze(-1)
        if play_indices.numel() > 0:
            play_logits = play_log_probs[play_indices]
            play_actions = actions[play_indices]
            action_selector = play_actions.view(-1, 1, 1).expand(
                -1, 1, self.config.atoms
            )
            chosen = play_logits.gather(1, action_selector).squeeze(1)
            target = projected[play_indices]
            per_sample = -(target * chosen).sum(dim=-1)
            play_loss = (per_sample * weights[play_indices]).mean()
            priorities[play_indices] = per_sample.detach()
            loss_play_value = play_loss.detach()

        if bet_loss is not None:
            self.optimizer_bet.zero_grad(set_to_none=True)
        if play_loss is not None:
            self.optimizer_play.zero_grad(set_to_none=True)

        if bet_loss is not None:
            bet_loss.backward(retain_graph=play_loss is not None)
        if play_loss is not None:
            play_loss.backward()

        if (
            bet_loss is not None or play_loss is not None
        ) and self.config.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), self.config.grad_clip
            )

        if bet_loss is not None:
            self.optimizer_bet.step()
        if play_loss is not None:
            self.optimizer_play.step()

        self.replay.update_priorities(sample.indices, priorities.cpu().numpy() + 1e-6)

        self.training_steps += 1
        self._update_target_network()
        if self.config.noisy_nets:
            self.policy_net.reset_noise()

        return {
            "loss_play": float(loss_play_value),
            "loss_bet": float(loss_bet_value),
            "epsilon": float(self.epsilon()),
        }

    def _project_distribution(
        self,
        next_dist: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rewards.size(0)
        support = self.support.view(1, -1)
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        n_steps = n_steps.view(-1, 1)
        gamma_n = torch.pow(self.config.gamma, n_steps)
        tz = rewards + (1.0 - dones) * gamma_n * support
        tz = tz.clamp(self.config.v_min, self.config.v_max)
        b = (tz - self.config.v_min) / self.delta_z
        lower_idx = b.floor().to(torch.int64)
        upper_idx = b.ceil().to(torch.int64)
        lower_idx = lower_idx.clamp(0, self.config.atoms - 1)
        upper_idx = upper_idx.clamp(0, self.config.atoms - 1)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.config.atoms, batch_size, device=self.device
            )
            .unsqueeze(1)
            .expand(batch_size, self.config.atoms)
            .to(torch.int64)
        )
        projected = torch.zeros(batch_size * self.config.atoms, device=self.device)
        projected.index_add_(
            0,
            (lower_idx + offset).view(-1),
            (next_dist * (upper_idx.float() - b)).view(-1),
        )
        projected.index_add_(
            0,
            (upper_idx + offset).view(-1),
            (next_dist * (b - lower_idx.float())).view(-1),
        )
        return projected.view(batch_size, self.config.atoms)

    def _update_target_network(self) -> None:
        if self.config.tau is not None:
            tau = self.config.tau
            for target_param, param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
        elif (
            self.config.target_update_interval
            and self.training_steps % self.config.target_update_interval == 0
        ):
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer_play": self.optimizer_play.state_dict(),
                "optimizer_bet": self.optimizer_bet.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["policy"])
        self.target_net.load_state_dict(data["target"])
        self.optimizer_play.load_state_dict(data["optimizer_play"])
        self.optimizer_bet.load_state_dict(data["optimizer_bet"])


__all__ = [
    "BasicStrategyAgent",
    "TabularQLearningAgent",
    "TabularConfig",
    "DQNConfig",
    "RainbowDQNAgent",
]
