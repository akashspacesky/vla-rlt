"""
Off-policy replay buffer for RLT training.

Stores transitions as (rl_token, vla_action, delta_action, reward, next_rl_token, done).
Note: we store pre-computed RL tokens rather than raw observations to avoid re-running
the GR00T encoder on every sample — critical for fast on-robot training.

Supports:
- Human intervention injection: mark demonstrations as high-reward for priority sampling
- Prioritized sampling (optional): sample high-reward transitions more often early in training
"""

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class Transition:
    rl_token: np.ndarray         # [d_rlt]
    vla_action: np.ndarray       # [chunk_size, action_dim]
    delta_action: np.ndarray     # [chunk_size, action_dim]
    reward: float
    next_rl_token: np.ndarray    # [d_rlt]
    done: bool
    is_human_demo: bool = False  # High-priority human intervention


class ReplayBuffer:
    """
    Circular replay buffer. Stores up to `capacity` transitions.
    Human demonstrations are stored separately and always included in batches
    at a configurable ratio.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        d_rlt: int = 256,
        action_dim: int = 7,
        chunk_size: int = 16,
        human_demo_ratio: float = 0.25,  # Fraction of each batch from human demos
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.d_rlt = d_rlt
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.human_demo_ratio = human_demo_ratio
        self.device = device

        # Pre-allocate arrays
        self.rl_tokens = np.zeros((capacity, d_rlt), dtype=np.float32)
        self.next_rl_tokens = np.zeros((capacity, d_rlt), dtype=np.float32)
        self.vla_actions = np.zeros((capacity, chunk_size, action_dim), dtype=np.float32)
        self.delta_actions = np.zeros((capacity, chunk_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Separate storage for human demos (never overwritten)
        self.human_rl_tokens: list[np.ndarray] = []
        self.human_next_rl_tokens: list[np.ndarray] = []
        self.human_vla_actions: list[np.ndarray] = []
        self.human_delta_actions: list[np.ndarray] = []
        self.human_rewards: list[float] = []
        self.human_dones: list[float] = []

        self.ptr = 0
        self.size = 0

    def add(
        self,
        rl_token: np.ndarray | torch.Tensor,
        vla_action: np.ndarray | torch.Tensor,
        delta_action: np.ndarray | torch.Tensor,
        reward: float,
        next_rl_token: np.ndarray | torch.Tensor,
        done: bool,
        is_human_demo: bool = False,
    ):
        """Add a transition to the buffer."""
        # Convert tensors to numpy
        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.array(x, dtype=np.float32)

        rl_token = to_np(rl_token)
        vla_action = to_np(vla_action)
        delta_action = to_np(delta_action)
        next_rl_token = to_np(next_rl_token)

        if is_human_demo:
            # Human demos go in separate persistent storage
            self.human_rl_tokens.append(rl_token)
            self.human_next_rl_tokens.append(next_rl_token)
            self.human_vla_actions.append(vla_action)
            self.human_delta_actions.append(delta_action)
            self.human_rewards.append(float(reward))
            self.human_dones.append(float(done))
        else:
            # RL transitions go in circular buffer
            self.rl_tokens[self.ptr] = rl_token
            self.next_rl_tokens[self.ptr] = next_rl_token
            self.vla_actions[self.ptr] = vla_action
            self.delta_actions[self.ptr] = delta_action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = float(done)
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Sample a batch of transitions. If human demos are available, include
        `human_demo_ratio` fraction from them.
        """
        has_human = len(self.human_rl_tokens) > 0
        n_human = int(batch_size * self.human_demo_ratio) if has_human else 0
        n_rl = batch_size - n_human

        def _batch_from_indices(indices, from_human: bool) -> dict[str, np.ndarray]:
            if from_human:
                h_idx = np.array(indices)
                return {
                    "rl_tokens": np.stack([self.human_rl_tokens[i] for i in h_idx]),
                    "next_rl_tokens": np.stack([self.human_next_rl_tokens[i] for i in h_idx]),
                    "vla_actions": np.stack([self.human_vla_actions[i] for i in h_idx]),
                    "delta_actions": np.stack([self.human_delta_actions[i] for i in h_idx]),
                    "rewards": np.array([self.human_rewards[i] for i in h_idx]),
                    "dones": np.array([self.human_dones[i] for i in h_idx]),
                }
            else:
                return {
                    "rl_tokens": self.rl_tokens[indices],
                    "next_rl_tokens": self.next_rl_tokens[indices],
                    "vla_actions": self.vla_actions[indices],
                    "delta_actions": self.delta_actions[indices],
                    "rewards": self.rewards[indices],
                    "dones": self.dones[indices],
                }

        # Sample RL transitions
        rl_indices = np.random.randint(0, self.size, size=n_rl)
        rl_batch = _batch_from_indices(rl_indices, from_human=False)

        # Sample and merge human demos
        if n_human > 0:
            human_indices = np.random.randint(0, len(self.human_rl_tokens), size=n_human)
            h_batch = _batch_from_indices(human_indices, from_human=True)
            combined = {
                k: np.concatenate([rl_batch[k], h_batch[k]], axis=0)
                for k in rl_batch
            }
        else:
            combined = rl_batch

        # Convert to tensors
        return {
            k: torch.tensor(v, dtype=torch.float32).to(self.device)
            for k, v in combined.items()
        }

    def __len__(self) -> int:
        return self.size

    @property
    def num_human_demos(self) -> int:
        return len(self.human_rl_tokens)

    def is_ready(self, min_size: int = 1000) -> bool:
        return self.size >= min_size
