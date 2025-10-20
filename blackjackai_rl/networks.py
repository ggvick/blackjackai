"""Neural network components for Rainbow DQN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer."""

    def __init__(
        self, in_features: int, out_features: int, sigma_init: float = 0.5
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features**0.5)
        self.bias_sigma.data.fill_(self.sigma_init / self.out_features**0.5)

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(input, weight, bias)


@dataclass
class RainbowHeadConfig:
    in_features: int
    num_actions: int
    atoms: int
    noisy: bool
    dueling: bool


class RainbowHead(nn.Module):
    def __init__(self, config: RainbowHeadConfig) -> None:
        super().__init__()
        Linear = NoisyLinear if config.noisy else nn.Linear
        self.atoms = config.atoms
        self.num_actions = config.num_actions
        hidden = config.in_features
        if config.dueling:
            self.value = Linear(hidden, config.atoms)
            self.advantage = Linear(hidden, config.num_actions * config.atoms)
        else:
            self.value = None
            self.advantage = Linear(hidden, config.num_actions * config.atoms)
        self.dueling = config.dueling
        self.noisy = config.noisy

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        if self.dueling and self.value is not None:
            value = self.value(features).view(batch_size, 1, self.atoms)
            advantage = self.advantage(features).view(
                batch_size, self.num_actions, self.atoms
            )
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            logits = self.advantage(features)
            q_atoms = logits.view(batch_size, self.num_actions, self.atoms)
        return q_atoms

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        if self.value is not None:
            assert isinstance(self.value, NoisyLinear)
            self.value.reset_noise()
        if isinstance(self.advantage, NoisyLinear):
            self.advantage.reset_noise()


class RainbowDQN(nn.Module):
    """Dual-head Rainbow DQN network with shared feature trunk."""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Sequence[int],
        bet_actions: int,
        play_actions: int,
        atoms: int,
        noisy: bool,
        dueling: bool,
    ) -> None:
        super().__init__()
        Linear = NoisyLinear if noisy else nn.Linear
        layers: List[nn.Module] = []
        input_dim = state_dim
        for hidden in hidden_sizes:
            layers.append(Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            input_dim = hidden
        self.feature_layers = nn.Sequential(*layers)
        head_config = RainbowHeadConfig(
            in_features=input_dim,
            num_actions=play_actions,
            atoms=atoms,
            noisy=noisy,
            dueling=dueling,
        )
        bet_head_config = RainbowHeadConfig(
            in_features=input_dim,
            num_actions=bet_actions,
            atoms=atoms,
            noisy=noisy,
            dueling=dueling,
        )
        self.play_head = RainbowHead(head_config)
        self.bet_head = RainbowHead(bet_head_config)
        self.noisy = noisy

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = state
        for layer in self.feature_layers:
            features = layer(features)
        play_atoms = self.play_head(features)
        bet_atoms = self.bet_head(features)
        return bet_atoms, play_atoms

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for layer in self.feature_layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        self.play_head.reset_noise()
        self.bet_head.reset_noise()


__all__ = ["RainbowDQN", "NoisyLinear"]
