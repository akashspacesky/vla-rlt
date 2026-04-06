"""
RLT Actor: Learns to edit VLA actions rather than replace them.

Key design choice from Pi's RLT paper: the actor receives the VLA's predicted
action chunk as input and outputs a *delta* (additive correction). This anchors
RL exploration close to the base VLA policy, preventing catastrophic forgetting
and making the search space tractable.

Reference-action dropout: with probability `ref_dropout_prob`, the VLA reference
action is zeroed out during forward pass, forcing the actor to learn an independent
pathway. This is critical early in training when the reference action dominates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class RLTActor(nn.Module):
    """
    Stochastic actor for SAC. Takes (rl_token, vla_action_chunk) and outputs
    a distribution over delta_action. The final executed action is:

        action = vla_action + tanh(delta_action) * action_scale

    Using tanh squashing keeps deltas bounded and prevents the actor from
    diverging too far from the VLA reference.
    """

    def __init__(
        self,
        d_rlt: int = 256,
        action_dim: int = 7,        # DoF per timestep
        chunk_size: int = 16,       # Action chunk length
        hidden_dim: int = 512,
        num_layers: int = 3,
        ref_dropout_prob: float = 0.3,
        action_scale: float = 0.1,  # Max delta magnitude (tune per task)
    ):
        super().__init__()
        self.d_rlt = d_rlt
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.ref_dropout_prob = ref_dropout_prob
        self.action_scale = action_scale

        # Flatten action chunk for input: chunk_size * action_dim
        action_flat_dim = chunk_size * action_dim

        # Input: [rl_token | vla_action_flat]
        input_dim = d_rlt + action_flat_dim

        # Build MLP trunk
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Mean and log-std heads for delta distribution
        self.mean_head = nn.Linear(hidden_dim, action_flat_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_flat_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Last layer small init to start near zero delta
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.log_std_head.weight)

    def forward(
        self,
        rl_token: torch.Tensor,
        vla_action_chunk: torch.Tensor,
        deterministic: bool = False,
        return_log_prob: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            rl_token: [batch, d_rlt]
            vla_action_chunk: [batch, chunk_size, action_dim] — VLA reference action
            deterministic: use mean instead of sampling (for eval)
            return_log_prob: compute log_prob for SAC loss

        Returns:
            dict with 'action_chunk', 'delta', 'log_prob', 'mean', 'log_std'
        """
        batch = rl_token.shape[0]
        action_flat = vla_action_chunk.reshape(batch, -1)  # [B, chunk*action_dim]

        # Reference-action dropout: randomly zero the VLA reference
        # Forces actor to learn action from RL token alone
        if self.training and self.ref_dropout_prob > 0:
            mask = (torch.rand(batch, 1, device=rl_token.device) > self.ref_dropout_prob).float()
            action_flat = action_flat * mask

        # Concatenate rl_token and (possibly dropped) VLA action
        x = torch.cat([rl_token, action_flat], dim=-1)  # [B, d_rlt + chunk*action_dim]

        # MLP trunk
        features = self.trunk(x)  # [B, hidden_dim]

        # Distribution parameters
        mean = self.mean_head(features)                                  # [B, chunk*action_dim]
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)

        if deterministic:
            delta_flat = mean
        else:
            delta_flat = dist.rsample()  # Reparameterized sample for SAC

        # Tanh squash the delta
        delta_squashed = torch.tanh(delta_flat) * self.action_scale

        # Reshape back to chunk
        delta = delta_squashed.reshape(batch, self.chunk_size, self.action_dim)

        # Final action: VLA reference + delta
        final_action = vla_action_chunk + delta

        result = {
            "action_chunk": final_action,  # [B, chunk_size, action_dim]
            "delta": delta,                # [B, chunk_size, action_dim]
            "mean": mean.reshape(batch, self.chunk_size, self.action_dim),
            "log_std": log_std.reshape(batch, self.chunk_size, self.action_dim),
        }

        if return_log_prob:
            # Log prob with tanh correction: log_pi(a) = log_pi(u) - sum(log(1 - tanh(u)^2))
            log_prob = dist.log_prob(delta_flat)
            log_prob -= torch.log(self.action_scale * (1 - delta_squashed.pow(2) / self.action_scale**2) + 1e-6)
            result["log_prob"] = log_prob.sum(dim=-1)  # [B]

        return result

    def get_action(
        self,
        rl_token: torch.Tensor,
        vla_action_chunk: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Convenience method returning only the final action chunk."""
        with torch.no_grad():
            result = self.forward(rl_token, vla_action_chunk, deterministic=deterministic, return_log_prob=False)
        return result["action_chunk"]

    def set_ref_dropout_prob(self, prob: float):
        """Anneal reference-action dropout probability during training."""
        self.ref_dropout_prob = prob
