"""Rainbow-style DQN agent for Blackjack."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # PyTorch 2.4+
    from torch.amp import GradScaler as _GradScaler  # type: ignore[attr-defined]
    _GRAD_SCALER_SUPPORTS_DEVICE = True
except ImportError:  # pragma: no cover - compatibility for older PyTorch
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore[attr-defined]
    _GRAD_SCALER_SUPPORTS_DEVICE = False

GradScaler = _GradScaler

from blackjack_env.masking import Action, apply_action_mask

from .replay import PrioritizedReplayBuffer
from .utils_device import get_device
from .gpu_utils import (
    autocast_if,
    maybe_compile,
    safe_state_dict_from_module,
    strip_orig_mod_prefix,
)


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
        mu_range = 1.0 / np.sqrt(self.in_features)
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


def mask_q(q: torch.Tensor, legal: torch.Tensor, neg_inf: float = float("-1e9")) -> torch.Tensor:
    """Apply legality mask to Q-values without in-place modification."""

    return q.masked_fill(~legal, neg_inf)


def normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    den = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (probs / den).clamp(1e-6, 1.0)


def _to_device(batch: Dict[str, torch.Tensor | np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        tensor: torch.Tensor
        if isinstance(value, torch.Tensor):
            tensor = value.to(device, non_blocking=True)
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).to(device)
        else:
            tensor = torch.as_tensor(value, device=device)
        out[key] = tensor
    return out


@dataclass
class AgentConfig:
    observation_dim: int
    bet_actions: int
    play_actions: int = 5
    hidden_sizes: Tuple[int, ...] = (1024, 1024)
    lr: float = 3e-4
    wd: float = 0.0
    buffer_size: int = 1_000_000
    batch_size: int = 1024
    min_buffer_size: int = 20_000
    target_update_interval: int = 15_000
    double_dqn: bool = True
    dueling: bool = True
    prioritized_replay: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_increment: float | None = None
    per_beta_steps: int = 1_000_000
    n_step: int = 3
    enable_c51: bool = True
    num_atoms: int = 51
    vmin: float = -20.0
    vmax: float = 20.0
    use_noisy: bool = True
    epsilon_start: float = 0.0
    epsilon_final: float = 0.0
    epsilon_decay: int = 0
    use_amp: bool = True
    replay_on_gpu: bool = True
    compile_model: bool = True
    gamma: float = 0.99
    clip_reward: float = 5.0
    detect_anomaly: bool = False
    grad_clip: float | None = 5.0
    device: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.hidden_sizes, Iterable) and not isinstance(
            self.hidden_sizes, tuple
        ):
            self.hidden_sizes = tuple(self.hidden_sizes)
        if self.per_beta_increment is None:
            steps = max(1, int(self.per_beta_steps))
            self.per_beta_increment = (self.per_beta_end - self.per_beta_start) / steps
        self.per_beta_increment = float(max(0.0, self.per_beta_increment))
        self.num_atoms = int(self.num_atoms)
        self.atom_size = self.num_atoms  # backward compatibility


@dataclass
class TrainConfig:
    steps: int = 2_000_000
    vector_envs: int = 64
    log_interval: int = 2000
    eval_hands: int = 100_000


class RainbowNet(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        in_dim = cfg.observation_dim
        for hidden in cfg.hidden_sizes:
            layers.append(linear(in_dim, hidden, cfg.use_noisy))
            layers.append(nn.ReLU())
            in_dim = hidden
        self.feature = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = in_dim if layers else cfg.observation_dim
        stream_hidden = max(last_dim // 2, 1)
        value_out = cfg.num_atoms if cfg.enable_c51 else 1
        advantage_out = cfg.play_actions * (cfg.num_atoms if cfg.enable_c51 else 1)
        if cfg.dueling:
            self.value_stream = nn.Sequential(
                linear(last_dim, stream_hidden, cfg.use_noisy),
                nn.ReLU(),
                linear(stream_hidden, value_out, cfg.use_noisy),
            )
            self.advantage_stream = nn.Sequential(
                linear(last_dim, stream_hidden, cfg.use_noisy),
                nn.ReLU(),
                linear(stream_hidden, advantage_out, cfg.use_noisy),
            )
            self.q_stream = None
        else:
            self.value_stream = None
            self.advantage_stream = None
            self.q_stream = nn.Sequential(
                linear(last_dim, stream_hidden, cfg.use_noisy),
                nn.ReLU(),
                linear(stream_hidden, advantage_out, cfg.use_noisy),
            )
        self.bet_head = nn.Sequential(
            linear(last_dim, stream_hidden, cfg.use_noisy),
            nn.ReLU(),
            linear(stream_hidden, cfg.bet_actions, cfg.use_noisy),
        )
        support = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_atoms)
        self.register_buffer("support", support)
        self.enable_c51 = cfg.enable_c51
        self.num_atoms = cfg.num_atoms
        self.play_actions = cfg.play_actions

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature(obs)
        bet_logits = self.bet_head(features)
        if self.cfg.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            if self.enable_c51:
                value = value.view(-1, 1, self.num_atoms)
                advantage = advantage.view(-1, self.play_actions, self.num_atoms)
            else:
                value = value.view(-1, 1)
                advantage = advantage.view(-1, self.play_actions)
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q = self.q_stream(features)
            if self.enable_c51:
                q = q.view(-1, self.play_actions, self.num_atoms)
            else:
                q = q.view(-1, self.play_actions)
        return bet_logits, q

    def play_Q(self, obs: torch.Tensor) -> torch.Tensor:
        _, q = self.forward(obs)
        if self.enable_c51:
            probs = torch.softmax(q, dim=-1)
            probs = normalize_probs(probs)
            return (probs * self.support.view(1, 1, -1)).sum(dim=-1)
        return q

    def reset_noise(self) -> None:
        if not self.cfg.use_noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class RainbowDQNAgent:
    """Encapsulates Rainbow DQN training logic."""

    def __init__(self, cfg: AgentConfig):
        self.config = cfg
        self.device = torch.device(cfg.device) if cfg.device else get_device()
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but not available. In Colab: Runtime → Change runtime type → GPU."
            )
        if self.config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        if self.config.use_noisy:
            self.config.epsilon_decay = 0
        self.online = RainbowNet(cfg).to(self.device)
        self.target = RainbowNet(cfg).to(self.device)
        self._sync_target_network()
        compile_enabled = bool(self.config.compile_model)
        self.online = maybe_compile(self.online, enabled=compile_enabled)
        self._sync_target_network()
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=cfg.lr, weight_decay=cfg.wd
        )
        self.global_step = 0
        self.amp_enabled = bool(self.config.use_amp and self.device.type == "cuda")
        scaler_kwargs = {"enabled": self.amp_enabled}
        if _GRAD_SCALER_SUPPORTS_DEVICE and self.device.type == "cuda":
            scaler_kwargs["device"] = "cuda"
        self.scaler = GradScaler(**scaler_kwargs)
        buffer_device = self.device if (cfg.replay_on_gpu and self.device.type == "cuda") else torch.device("cpu")
        self.buffer = PrioritizedReplayBuffer(
            capacity=cfg.buffer_size,
            observation_dim=cfg.observation_dim,
            action_dim=cfg.play_actions,
            device=buffer_device,
            alpha=cfg.per_alpha if cfg.prioritized_replay else 0.0,
            beta_start=cfg.per_beta_start,
            beta_end=cfg.per_beta_end,
            beta_increment=cfg.per_beta_increment,
            use_amp=self.amp_enabled,
            replay_on_gpu=cfg.replay_on_gpu and self.device.type == "cuda",
        )
        self.n_step_buffer: Deque[Tuple] = deque(maxlen=cfg.n_step)
        self.delta_z = (
            (cfg.vmax - cfg.vmin) / (cfg.num_atoms - 1)
            if cfg.enable_c51 and cfg.num_atoms > 1
            else None
        )
        self.gamma_n = self.config.gamma ** self.config.n_step
        self._maybe_reset_noise()

    # ------------------------------------------------------------------
    def _sync_target_network(self) -> None:
        state = safe_state_dict_from_module(self.online)
        self.target.load_state_dict(state)

    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        if self.config.use_noisy:
            return 0.0
        if self.config.epsilon_decay <= 0:
            return self.config.epsilon_final
        fraction = min(self.global_step / self.config.epsilon_decay, 1.0)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_final - self.config.epsilon_start
        )

    def _maybe_reset_noise(self) -> None:
        if self.config.use_noisy:
            self.online.reset_noise()
            self.target.reset_noise()

    def act_bet(self, observation: np.ndarray) -> int:
        obs = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self._maybe_reset_noise()
            bet_logits, _ = self.online(obs)
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
            q_values = self.online.play_Q(obs).squeeze(0).cpu().numpy()
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
    def _loss_from_batch(
        self, batch: Dict[str, torch.Tensor], weights: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = batch["states"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        legal = batch["legal_mask"].to(self.device).bool()
        legal_next = batch["legal_mask_next"].to(self.device).bool()
        bet_actions = batch["bet_actions"].to(self.device).long()
        actions = batch["actions"].to(self.device).long()
        rewards = batch["rewards"].to(self.device, dtype=torch.float32)
        dones = batch["dones"].to(self.device, dtype=torch.float32)

        self._maybe_reset_noise()
        bet_logits, play_outputs = self.online(states)
        if self.config.enable_c51:
            support_values = self.target.support.to(self.device)
            support = support_values.view(1, 1, -1)
            online_logits = play_outputs
            log_probs = F.log_softmax(online_logits, dim=-1)
            probs = normalize_probs(torch.softmax(online_logits, dim=-1))
            q_expectation = (probs * support).sum(dim=-1)
            q_masked = mask_q(q_expectation, legal)
            q_sa = q_masked.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_online_logits = self.online(next_states)[1]
                next_online_probs = normalize_probs(
                    torch.softmax(next_online_logits, dim=-1)
                )
                q2o = mask_q((next_online_probs * support).sum(dim=-1), legal_next)
                a2 = q2o.argmax(dim=1)
                next_target_logits = self.target(next_states)[1]
                target_probs_all = normalize_probs(
                    torch.softmax(next_target_logits, dim=-1)
                )
                gather_index = a2.view(-1, 1, 1).expand(-1, 1, self.config.num_atoms)
                target_probs = target_probs_all.gather(1, gather_index).squeeze(1)
                tz = rewards.unsqueeze(-1) + self.gamma_n * (1.0 - dones.unsqueeze(-1)) * support_values.view(1, -1)
                tz = tz.clamp(self.config.vmin, self.config.vmax)
                b = (tz - self.config.vmin) / self.delta_z if self.delta_z else tz
                l = b.floor()
                u = b.ceil()
                l_idx = l.clamp(0, self.config.num_atoms - 1).long()
                u_idx = u.clamp(0, self.config.num_atoms - 1).long()
                m = torch.zeros_like(target_probs)
                m.scatter_add_(dim=-1, index=l_idx, src=target_probs * (u - b))
                m.scatter_add_(dim=-1, index=u_idx, src=target_probs * (b - l))
            action_log_probs = log_probs.gather(
                1, actions.view(-1, 1, 1).expand(-1, 1, self.config.num_atoms)
            ).squeeze(1)
            q_loss = -(m * action_log_probs).sum(dim=-1)
            bet_targets = (m * support_values.view(1, -1)).sum(dim=-1)
        else:
            q_values = play_outputs
            q_masked = mask_q(q_values, legal)
            q_sa = q_masked.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next_online = mask_q(self.online.play_Q(next_states), legal_next)
                a2 = q_next_online.argmax(dim=1)
                q_next_target = mask_q(self.target.play_Q(next_states), legal_next)
                q_next = q_next_target.gather(1, a2.unsqueeze(1)).squeeze(1)
                bet_targets = rewards + self.gamma_n * (1.0 - dones) * q_next
            q_loss = F.smooth_l1_loss(q_sa, bet_targets, reduction="none")

        bet_pred = bet_logits.gather(1, bet_actions.unsqueeze(1)).squeeze(1)
        bet_loss = F.mse_loss(bet_pred, bet_targets, reduction="none")
        per_sample = q_loss + bet_loss
        if weights is not None:
            per_sample = per_sample * weights
        loss = per_sample.mean()
        td_error = (bet_targets - q_sa).abs()
        return loss, per_sample.detach(), td_error.detach(), bet_targets.detach()

    def train_step(self) -> Dict[str, float | None]:
        if self.buffer.pos < self.config.min_buffer_size and not self.buffer.full:
            self.global_step += 1
            return {"loss": None}
        batch, indices, weights_np = self.buffer.sample(self.config.batch_size)
        batch_t = _to_device(batch, self.device)
        isw = (
            torch.as_tensor(weights_np, device=self.device)
            if weights_np is not None
            else None
        )
        if isw is not None:
            isw = isw.clamp_min(1e-3)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast_if(self.amp_enabled):
            loss, _, td_error, _ = self._loss_from_batch(batch_t, isw)
        if self.amp_enabled:
            self.scaler.scale(loss).backward()
            if self.config.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.online.parameters(), self.config.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.online.parameters(), self.config.grad_clip
                )
            self.optimizer.step()
        self.buffer.update_priorities(indices, td_error.abs().cpu().numpy() + 1e-6)
        if self.global_step % self.config.target_update_interval == 0:
            self._sync_target_network()
        loss_value = float(loss.detach().cpu())
        self.global_step += 1
        return {"loss": loss_value}

    # ------------------------------------------------------------------
    def _debug_train_step_once(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_t = _to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        isw = batch_t.get("weights")
        if isw is not None:
            batch_t = {k: v for k, v in batch_t.items() if k != "weights"}
            weights = isw.clamp_min(1e-3)
        else:
            weights = None
        with autocast_if(self.amp_enabled):
            loss, _, _, _ = self._loss_from_batch(batch_t, weights)
        if self.amp_enabled:
            self.scaler.scale(loss).backward()
            if self.config.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.online.parameters(), self.config.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.online.parameters(), self.config.grad_clip
                )
            self.optimizer.step()
        return loss.detach()

    # ------------------------------------------------------------------
    def train(self, env, steps: int, callback=None) -> Dict[str, float | None]:
        metrics = {"loss": None}
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
            next_mask = (
                info.get("mask", env.available_actions())
                if not done
                else np.zeros(self.config.play_actions, dtype=bool)
            )
            clipped_reward = float(
                np.clip(reward, -self.config.clip_reward, self.config.clip_reward)
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
            metrics = self.train_step()
            if done:
                observation = env.reset()
                mask = env.available_actions()
                bet_action = self.act_bet(observation)
                observation, _, done, info = env.step({"bet": bet_action})
                mask = (
                    info.get("mask", env.available_actions())
                    if not done
                    else np.zeros(self.config.play_actions, dtype=bool)
                )
            if callback and step % 1000 == 0:
                callback(step, metrics)
        return metrics

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a state_dict that is safe for compiled models."""

        return safe_state_dict_from_module(self.online)

    # ------------------------------------------------------------------
    def load_weights(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        """Load weights into the online and target networks safely."""

        cleaned = strip_orig_mod_prefix(state_dict)
        base_online = getattr(self.online, "_orig_mod", self.online)
        base_online.load_state_dict(cleaned, strict=strict)
        self.target.load_state_dict(cleaned, strict=strict)
        self._maybe_reset_noise()

    # ------------------------------------------------------------------
    def load_checkpoint(self, payload: Dict[str, Any], strict: bool = True) -> None:
        """Load a checkpoint payload with architecture safety checks."""

        if not isinstance(payload, dict):
            raise TypeError("Checkpoint payload must be a dictionary")
        config_section = payload.get("config")
        saved_agent_cfg: Dict[str, Any] = {}
        if isinstance(config_section, dict):
            agent_cfg = config_section.get("agent")
            if isinstance(agent_cfg, dict):
                saved_agent_cfg = agent_cfg
        mismatches = []
        for key in ("use_noisy", "enable_c51"):
            if key in saved_agent_cfg and saved_agent_cfg[key] != getattr(self.config, key):
                mismatches.append(key)
        if mismatches:
            raise RuntimeError(
                "Checkpoint architecture mismatch: use_noisy/enable_c51 differ. "
                "Recreate agent with matching flags."
            )
        state = payload.get("model", payload)
        if not isinstance(state, dict):
            raise TypeError("Checkpoint payload missing model state_dict")
        self.load_weights(state, strict=strict)
        if "step" in payload:
            try:
                self.global_step = int(payload["step"])
            except (TypeError, ValueError):
                pass
