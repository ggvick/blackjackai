"""Rainbow-style DQN agent for Blackjack."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blackjack_env.masking import Action, apply_action_mask

from .replay import PrioritizedReplayBuffer


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.sigma0 = sigma0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma0 / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma0 / np.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.training:
            eps_in = self._scale_noise(self.in_features)
            eps_out = self._scale_noise(self.out_features)
            self.weight_eps.copy_(eps_out.ger(eps_in))
            self.bias_eps.copy_(eps_out)
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


def linear(in_features: int, out_features: int, use_noisy: bool) -> nn.Module:
    if use_noisy:
        return NoisyLinear(in_features, out_features)
    return nn.Linear(in_features, out_features)


@dataclass
class AgentConfig:
    observation_dim: int
    bet_actions: int
    play_actions: int = 5
    atom_size: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    gamma: float = 0.99
    n_step: int = 3
    buffer_size: int = 1_000_000
    min_buffer_size: int = 20_000
    batch_size: int = 512
    lr: float = 1e-4
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 1e-6
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 1_200_000
    use_noisy: bool = False
    target_update_interval: int = 15000
    grad_clip: float = 5.0
    device: str = "cpu"
    use_amp: bool = True
    enable_c51: bool = True
    clip_reward: float = 5.0


@dataclass
class TrainConfig:
    steps: int = 2_000_000
    eval_interval: int = 50_000
    log_interval: int = 2000


class RainbowNetwork(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.feature = nn.Sequential(
            nn.Linear(config.observation_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        value_out = config.atom_size if config.enable_c51 else 1
        advantage_out = (
            (config.play_actions * config.atom_size)
            if config.enable_c51
            else config.play_actions
        )
        value_stream = [
            linear(512, 512, config.use_noisy),
            nn.ReLU(),
            linear(512, value_out, config.use_noisy),
        ]
        advantage_stream = [
            linear(512, 512, config.use_noisy),
            nn.ReLU(),
            linear(512, advantage_out, config.use_noisy),
        ]
        self.value_stream = nn.Sequential(*value_stream)
        self.advantage_stream = nn.Sequential(*advantage_stream)
        self.bet_head = nn.Sequential(
            linear(512, 256, config.use_noisy),
            nn.ReLU(),
            linear(256, config.bet_actions, config.use_noisy),
        )
        self.atom_size = config.atom_size
        self.play_actions = config.play_actions
        self.enable_c51 = config.enable_c51

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature(obs)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        if self.enable_c51:
            value = value.view(-1, 1, self.atom_size)
            advantage = advantage.view(-1, self.play_actions, self.atom_size)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            value = value.view(-1, 1)
            advantage = advantage.view(-1, self.play_actions)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        bet_logits = self.bet_head(features)
        return bet_logits, q_atoms


class RainbowDQNAgent:
    """Encapsulates Rainbow DQN training logic."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.online_net = RainbowNetwork(config).to(self.device)
        self.target_net = RainbowNetwork(config).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.buffer = PrioritizedReplayBuffer(
            config.buffer_size,
            alpha=config.per_alpha,
            beta=config.per_beta,
            beta_increment=config.per_beta_increment,
        )
        self.n_step_buffer: Deque[Tuple] = deque(maxlen=config.n_step)
        self.support = torch.linspace(
            config.v_min, config.v_max, config.atom_size, device=self.device
        )
        self.delta_z = (config.v_max - config.v_min) / (config.atom_size - 1)
        self.global_step = 0
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        if self.config.use_noisy:
            return 0.0
        fraction = min(self.global_step / self.config.epsilon_decay, 1.0)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def act_bet(self, observation: np.ndarray) -> int:
        obs = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            bet_logits, _ = self.online_net(obs)
            bet_action = int(bet_logits.argmax(dim=-1).item())
        return bet_action

    def act_play(
        self, observation: np.ndarray, mask: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        legal_indices = np.where(mask)[0]
        if legal_indices.size == 0:
            return Action.STAND, mask
        if np.random.rand() < self.epsilon():
            action = int(np.random.choice(legal_indices))
            q_values = np.zeros(mask.shape[0], dtype=np.float32)
            return action, q_values
        obs = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, q_atoms = self.online_net(obs)
            if self.config.enable_c51:
                probs = F.softmax(q_atoms, dim=-1)
                q_values = (probs * self.support).sum(dim=-1)
            else:
                q_values = q_atoms
            q_values = q_values.squeeze(0).cpu().numpy()
        masked_q = apply_action_mask(q_values, mask)
        action = int(masked_q.argmax())
        return action, q_values

    # ------------------------------------------------------------------
    def _push_transition(self, transition: Tuple) -> None:
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.config.n_step:
            return
        reward, next_state, done = 0.0, None, False
        for idx, trans in enumerate(self.n_step_buffer):
            s, bet_action, action, r, next_obs, d = trans
            reward += (self.config.gamma**idx) * r
            next_state = next_obs
            done = d
            if d:
                break
        first = self.n_step_buffer[0]
        state, bet_action, action, _, _, _ = first
        self.buffer.add((state, bet_action, action, reward, next_state, done))

    def store(self, state, bet_action, action, reward, next_state, done) -> None:
        self._push_transition((state, bet_action, action, reward, next_state, done))
        if done:
            self.n_step_buffer.clear()

    # ------------------------------------------------------------------
    def train_step(self) -> Dict[str, float]:
        if self.buffer.pos < self.config.min_buffer_size and not self.buffer.full:
            return {"loss": 0.0}
        transitions, indices, weights = self.buffer.sample(self.config.batch_size)
        states, bet_actions, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            np.stack(next_states), dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        bet_actions = torch.tensor(bet_actions, dtype=torch.long, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        weights_t = torch.tensor(
            weights, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            bet_logits, q_atoms = self.online_net(states)
            if self.config.enable_c51:
                q_probs = F.log_softmax(q_atoms, dim=-1)
                online_next = self.online_net(next_states)[1]
                target_next = self.target_net(next_states)[1]
                online_probs = F.softmax(online_next, dim=-1)
                target_probs = F.softmax(target_next, dim=-1)
                next_q_values = (online_probs * self.support).sum(dim=-1)
                next_actions = next_q_values.argmax(dim=-1)
                target_dist = target_probs[range(target_probs.size(0)), next_actions]
                tz = (
                    rewards.unsqueeze(-1)
                    + (self.config.gamma**self.config.n_step)
                    * (1 - dones.unsqueeze(-1))
                    * self.support
                )
                tz = tz.clamp(self.config.v_min, self.config.v_max)
                b = (tz - self.config.v_min) / self.delta_z
                lower_idx = b.floor().long()
                upper_idx = b.ceil().long()
                offset = (
                    torch.linspace(
                        0,
                        (self.config.batch_size - 1) * self.config.atom_size,
                        self.config.batch_size,
                        device=self.device,
                    )
                    .long()
                    .unsqueeze(1)
                )
                proj_dist = torch.zeros_like(target_dist)
                proj_dist.view(-1).index_add_(
                    0,
                    (lower_idx + offset).view(-1),
                    (target_dist * (upper_idx.float() - b)).view(-1),
                )
                proj_dist.view(-1).index_add_(
                    0,
                    (upper_idx + offset).view(-1),
                    (target_dist * (b - lower_idx.float())).view(-1),
                )
                action_log_probs = q_probs[range(q_probs.size(0)), actions]
                q_loss = -(proj_dist * action_log_probs).sum(dim=-1)
                target_q = (proj_dist * self.support).sum(dim=-1).detach()
            else:
                q_values = q_atoms
                q_selected = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                next_online = self.online_net(next_states)[1]
                next_target = self.target_net(next_states)[1]
                next_actions = next_online.argmax(dim=-1)
                next_q = next_target.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                target_q = (
                    rewards
                    + (self.config.gamma**self.config.n_step) * (1 - dones) * next_q
                )
                q_loss = F.smooth_l1_loss(
                    q_selected.unsqueeze(-1), target_q.unsqueeze(-1), reduction="none"
                ).squeeze(-1)
            bet_pred = bet_logits.gather(1, bet_actions.unsqueeze(-1)).squeeze(-1)
            bet_loss = F.mse_loss(bet_pred, target_q.detach(), reduction="none")
            loss = ((q_loss.unsqueeze(-1) + bet_loss.unsqueeze(-1)) * weights_t).mean()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.config.grad_clip
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        td_error = q_loss.detach().cpu().numpy()
        self.buffer.update_priorities(indices, np.abs(td_error) + 1e-6)

        if self.global_step % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {"loss": float(loss.item())}

    # ------------------------------------------------------------------
    def train(self, env, steps: int, callback=None) -> Dict[str, float]:
        metrics = {"loss": 0.0}
        observation = env.reset()
        mask = env.available_actions()
        bet_action = self.act_bet(observation)
        obs, reward, done, info = env.step({"bet": bet_action})
        if done:
            observation = env.reset()
        observation = obs
        for step in range(steps):
            if env.state.stage == "bet":
                bet_action = self.act_bet(observation)
                observation, reward, done, info = env.step({"bet": bet_action})
                if done:
                    observation = env.reset()
                    continue
            mask = env.available_actions()
            action, q_values = self.act_play(observation, mask)
            next_obs, reward, done, info = env.step(action)
            clipped_reward = np.clip(
                reward, -self.config.clip_reward, self.config.clip_reward
            )
            self.store(observation, bet_action, action, clipped_reward, next_obs, done)
            observation = next_obs
            bet_action = info.get("bet", bet_action)
            self.global_step += 1
            metrics = self.train_step()
            if done:
                observation = env.reset()
                bet_action = self.act_bet(observation)
                observation, _, done, _ = env.step({"bet": bet_action})
            if callback and step % 1000 == 0:
                callback(step, metrics)
        return metrics
