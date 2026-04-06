"""
SAC (Soft Actor-Critic) for RLT.

Key additions over vanilla SAC:
1. Reference regularization: actor loss includes L2 penalty toward VLA reference action,
   preventing the policy from drifting too far from the base VLA.
2. Action chunking: operates on full chunk_size action sequences.
3. Auto-entropy tuning: automatically adjusts temperature alpha.

Loss formulas:
    Critic:  L_Q = E[(Q(s,a) - (r + γ * (min_Q(s',a') - α*log π(a'|s'))))²]
    Actor:   L_π = E[α*log π(a|s) - min_Q(s,a) + λ * ||a - a_vla||²]
    Alpha:   L_α = E[-α * (log π(a|s) + H_target)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any
from copy import deepcopy

from groot_rlt.models.actor import RLTActor
from groot_rlt.models.critic import RLTCritic


class SAC:
    """
    SAC algorithm for RLT. Operates entirely in RL-token space:
    state = rl_token, action = delta_action_chunk (but actor also receives vla_action).
    """

    def __init__(
        self,
        actor: RLTActor,
        critic: RLTCritic,
        # Optimizer hyperparams
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        # SAC hyperparams
        gamma: float = 0.99,
        tau: float = 0.005,               # Soft target update rate
        init_alpha: float = 0.2,          # Initial entropy temperature
        target_entropy: float | None = None,  # Auto-computed if None
        # Reference regularization
        ref_reg_weight: float = 0.1,      # λ: weight of ||action - vla_action||² penalty
        # Training
        gradient_clip: float = 1.0,
        device: str = "cuda",
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.ref_reg_weight = ref_reg_weight
        self.gradient_clip = gradient_clip

        # Target critic (soft-updated, never directly trained)
        self.critic_target = deepcopy(critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Optimizers
        self.actor_optim = torch.optim.AdamW(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.AdamW(critic.parameters(), lr=critic_lr)

        # Auto-entropy tuning
        action_flat_dim = actor.chunk_size * actor.action_dim
        self.target_entropy = target_entropy or -action_flat_dim  # Heuristic: -|A|
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32,
                                       requires_grad=True, device=device)
        self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=alpha_lr)

        self.total_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        One SAC update step.

        Args:
            batch: dict from ReplayBuffer.sample() with keys:
                   rl_tokens, next_rl_tokens, vla_actions, delta_actions, rewards, dones

        Returns:
            dict of scalar losses for logging
        """
        rl_tokens = batch["rl_tokens"].to(self.device)
        next_rl_tokens = batch["next_rl_tokens"].to(self.device)
        vla_actions = batch["vla_actions"].to(self.device)
        rewards = batch["rewards"].to(self.device).unsqueeze(-1)  # [B, 1]
        dones = batch["dones"].to(self.device).unsqueeze(-1)       # [B, 1]

        # The action stored in buffer is the full final action (vla + delta)
        stored_actions = vla_actions + batch["delta_actions"].to(self.device)

        # ── Critic Update ──────────────────────────────────────────────────
        with torch.no_grad():
            # Sample next actions from actor
            next_actor_out = self.actor(
                next_rl_tokens,
                vla_actions,  # Use same VLA reference (no future VLA call needed)
                return_log_prob=True,
            )
            next_actions = next_actor_out["action_chunk"]
            next_log_prob = next_actor_out["log_prob"].unsqueeze(-1)  # [B, 1]

            # Bellman target
            q1_next, q2_next = self.critic_target(next_rl_tokens, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1, q2 = self.critic(rl_tokens, stored_actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optim.step()

        # ── Actor Update ───────────────────────────────────────────────────
        actor_out = self.actor(rl_tokens, vla_actions, return_log_prob=True)
        new_actions = actor_out["action_chunk"]
        log_prob = actor_out["log_prob"].unsqueeze(-1)  # [B, 1]

        q_new = self.critic.q_min(rl_tokens, new_actions)

        # Entropy-regularized actor loss
        sac_loss = (self.alpha.detach() * log_prob - q_new).mean()

        # Reference regularization: penalize deviation from VLA action
        ref_loss = F.mse_loss(new_actions, vla_actions)
        actor_loss = sac_loss + self.ref_reg_weight * ref_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.actor_optim.step()

        # ── Alpha Update ───────────────────────────────────────────────────
        alpha_loss = (-self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ── Soft Target Update ─────────────────────────────────────────────
        self._soft_update_target()

        self.total_updates += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "sac_loss": sac_loss.item(),
            "ref_loss": ref_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "log_prob_mean": log_prob.mean().item(),
            "q_mean": q_new.mean().item(),
        }

    def _soft_update_target(self):
        """Polyak averaging: θ_target = τ*θ + (1-τ)*θ_target"""
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, state: dict[str, Any]):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        self.alpha_optim.load_state_dict(state["alpha_optim"])
        self.log_alpha.data.fill_(np.log(state["log_alpha"]))
        self.total_updates = state["total_updates"]
