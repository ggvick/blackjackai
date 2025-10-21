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

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(torch.outer(eps_out, eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.training:
            self.reset_noise()
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


def mask_q(
    q: torch.Tensor, legal_mask: torch.Tensor, neg_inf: float = float("-1e9")
) -> torch.Tensor:
    """Apply a legality mask to Q-values without in-place modification."""

    return q.masked_fill(~legal_mask, neg_inf)


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
    detect_anomaly: bool = False


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

    def reset_noise(self) -> None:
        if not self.config.use_noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class RainbowDQNAgent:
    """Encapsulates Rainbow DQN training logic."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        if self.config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        if self.config.use_noisy:
            self.config.epsilon_decay = 0
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
        self.amp_enabled = bool(self.config.use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.amp_enabled)
        self._maybe_reset_noise()

    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        if self.config.use_noisy:
            return 0.0
        fraction = min(self.global_step / self.config.epsilon_decay, 1.0)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _maybe_reset_noise(self) -> None:
        if self.config.use_noisy:
            self.online_net.reset_noise()
            self.target_net.reset_noise()

    def act_bet(self, observation: np.ndarray) -> int:
        obs = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self._maybe_reset_noise()
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
            self._maybe_reset_noise()
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
        reward, next_state, next_mask, done = 0.0, None, None, False
        for idx, trans in enumerate(self.n_step_buffer):
            (
                _state,
                _mask,
                _bet_action,
                _action,
                r,
                next_obs,
                next_obs_mask,
                d,
            ) = trans
            reward += (self.config.gamma**idx) * r
            next_state = next_obs
            next_mask = next_obs_mask
            done = d
            if d:
                break
        first = self.n_step_buffer[0]
        state, mask, bet_action, action, _, _, _, _ = first
        self.buffer.add(
            (state, mask, bet_action, action, reward, next_state, next_mask, done)
        )

    def store(
        self,
        state,
        mask,
        bet_action,
        action,
        reward,
        next_state,
        next_mask,
        done,
    ) -> None:
        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = (
            np.asarray(next_state, dtype=np.float32)
            if next_state is not None
            else np.zeros(self.config.observation_dim, dtype=np.float32)
        )
        mask_arr = np.asarray(mask, dtype=bool)
        next_mask_arr = (
            np.asarray(next_mask, dtype=bool)
            if next_mask is not None
            else np.zeros(self.config.play_actions, dtype=bool)
        )
        self._push_transition(
            (
                state_arr,
                mask_arr,
                bet_action,
                action,
                reward,
                next_state_arr,
                next_mask_arr,
                done,
            )
        )
        if done:
            self.n_step_buffer.clear()

    # ------------------------------------------------------------------
    def train_step(self) -> Dict[str, float]:
        if self.buffer.pos < self.config.min_buffer_size and not self.buffer.full:
            return {"loss": 0.0}

        transitions, indices, weights = self.buffer.sample(self.config.batch_size)
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

        states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            np.stack(next_states), dtype=torch.float32, device=self.device
        )
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        bet_actions_t = torch.as_tensor(bet_actions, dtype=torch.long, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        legal_mask = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=self.device)
        legal_mask_next = torch.as_tensor(
            np.stack(next_masks), dtype=torch.bool, device=self.device
        )
        is_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        is_weights = is_weights.clamp_min(1e-3)
        gamma_n = self.config.gamma**self.config.n_step

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
            self._maybe_reset_noise()
            bet_logits, play_outputs = self.online_net(states_t)

            if self.config.enable_c51:
                online_logits = play_outputs
                log_probs = F.log_softmax(online_logits, dim=-1)
                probs = F.softmax(online_logits, dim=-1)
                support = self.support.view(1, 1, -1)
                q_expectation = (probs * support).sum(dim=-1)
                q_masked = mask_q(q_expectation, legal_mask)
                q_sa = q_masked.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_online_logits = self.online_net(next_states_t)[1]
                    next_target_logits = self.target_net(next_states_t)[1]
                    next_online_probs = F.softmax(next_online_logits, dim=-1)
                    next_q_expectation = (next_online_probs * support).sum(dim=-1)
                    next_q_masked = mask_q(next_q_expectation, legal_mask_next)
                    next_actions = next_q_masked.argmax(dim=-1)
                    target_all_probs = F.softmax(next_target_logits, dim=-1)
                    gather_index = next_actions.view(-1, 1, 1).expand(
                        -1, 1, self.config.atom_size
                    )
                    target_probs = target_all_probs.gather(1, gather_index).squeeze(1)
                    tz = rewards_t.unsqueeze(-1) + gamma_n * (
                        1.0 - dones_t.unsqueeze(-1)
                    ) * self.support
                    tz = tz.clamp(self.config.v_min, self.config.v_max)
                    b = (tz - self.config.v_min) / self.delta_z
                    l = b.floor()
                    u = b.ceil()
                    l_idx = l.clamp(0, self.config.atom_size - 1).long()
                    u_idx = u.clamp(0, self.config.atom_size - 1).long()
                    m = torch.zeros_like(target_probs)
                    m.scatter_add_(dim=-1, index=l_idx, src=target_probs * (u - b))
                    m.scatter_add_(dim=-1, index=u_idx, src=target_probs * (b - l))

                action_log_probs = log_probs.gather(
                    1,
                    actions_t.view(-1, 1, 1).expand(-1, 1, self.config.atom_size),
                ).squeeze(1)
                q_loss = -(m * action_log_probs).sum(dim=-1)
                bet_targets = (m * self.support).sum(dim=-1)
            else:
                q_values = play_outputs
                q_masked = mask_q(q_values, legal_mask)
                q_sa = q_masked.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_online = mask_q(
                        self.online_net(next_states_t)[1], legal_mask_next
                    )
                    next_actions = next_q_online.argmax(dim=1)
                    next_q_target = mask_q(
                        self.target_net(next_states_t)[1], legal_mask_next
                    )
                    q_next = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    bet_targets = rewards_t + gamma_n * (1.0 - dones_t) * q_next

                q_loss = F.smooth_l1_loss(q_sa, bet_targets, reduction="none")

            bet_pred = bet_logits.gather(1, bet_actions_t.unsqueeze(1)).squeeze(1)
            bet_loss = F.mse_loss(bet_pred, bet_targets, reduction="none")
            per_sample_loss = q_loss + bet_loss
            per_sample_loss = per_sample_loss * is_weights
            loss = per_sample_loss.mean()

        self.scaler.scale(loss).backward()
        if self.config.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.online_net.parameters(), self.config.grad_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        td_error = (bet_targets - q_sa).abs().detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_error + 1e-6)

        if self.global_step % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        loss_value = loss.detach().item()
        return {"loss": float(loss_value)}

    # ------------------------------------------------------------------
    def train(self, env, steps: int, callback=None) -> Dict[str, float]:
        metrics = {"loss": 0.0}
        observation = env.reset()
        mask = env.available_actions()
        bet_action = self.act_bet(observation)
        obs, reward, done, info = env.step({"bet": bet_action})
        if done:
            observation = env.reset()
            mask = env.available_actions()
        else:
            mask = info.get("mask", env.available_actions())
        observation = obs
        for step in range(steps):
            if env.state.stage == "bet":
                bet_action = self.act_bet(observation)
                observation, reward, done, info = env.step({"bet": bet_action})
                if done:
                    observation = env.reset()
                    mask = env.available_actions()
                    continue
            action, q_values = self.act_play(observation, mask)
            next_obs, reward, done, info = env.step(action)
            next_mask = info.get("mask", env.available_actions()) if not done else np.zeros(
                self.config.play_actions, dtype=bool
            )
            clipped_reward = np.clip(
                reward, -self.config.clip_reward, self.config.clip_reward
            )
            self.store(
                observation,
                mask,
                bet_action,
                action,
                clipped_reward,
                next_obs,
                next_mask,
                done,
            )
            observation = next_obs
            mask = next_mask
            bet_action = info.get("bet", bet_action)
            self.global_step += 1
            metrics = self.train_step()
            if done:
                observation = env.reset()
                mask = env.available_actions()
                bet_action = self.act_bet(observation)
                observation, _, done, info = env.step({"bet": bet_action})
                mask = info.get("mask", env.available_actions()) if not done else np.zeros(
                    self.config.play_actions, dtype=bool
                )
            if callback and step % 1000 == 0:
                callback(step, metrics)
        return metrics
