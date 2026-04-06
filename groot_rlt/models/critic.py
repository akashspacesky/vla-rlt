"""
RLT Twin-Q Critic for SAC.

Takes (rl_token, action_chunk) → Q-value. We use twin critics (Q1, Q2) and
take the minimum for the SAC target, which reduces overestimation bias.

The critic does NOT receive the VLA reference action — only the RL token
(state representation) and the action actually taken by the actor.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Single Q-network."""

    def __init__(
        self,
        d_rlt: int = 256,
        action_dim: int = 7,
        chunk_size: int = 16,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        action_flat_dim = chunk_size * action_dim
        input_dim = d_rlt + action_flat_dim

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, rl_token: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rl_token: [batch, d_rlt]
            action_chunk: [batch, chunk_size, action_dim]

        Returns:
            q_value: [batch, 1]
        """
        batch = rl_token.shape[0]
        action_flat = action_chunk.reshape(batch, -1)
        x = torch.cat([rl_token, action_flat], dim=-1)
        return self.net(x)


class RLTCritic(nn.Module):
    """
    Twin-Q critic. Standard SAC setup with two independent Q-networks.
    Uses minimum of Q1, Q2 for target computation.
    """

    def __init__(
        self,
        d_rlt: int = 256,
        action_dim: int = 7,
        chunk_size: int = 16,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        self.q1 = QNetwork(d_rlt, action_dim, chunk_size, hidden_dim, num_layers)
        self.q2 = QNetwork(d_rlt, action_dim, chunk_size, hidden_dim, num_layers)

    def forward(
        self,
        rl_token: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (q1, q2): both [batch, 1]
        """
        return self.q1(rl_token, action_chunk), self.q2(rl_token, action_chunk)

    def q_min(self, rl_token: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Min of twin Q-values. Used for actor loss."""
        q1, q2 = self.forward(rl_token, action_chunk)
        return torch.min(q1, q2)
