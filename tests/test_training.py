"""Unit tests for SAC training components."""

import pytest
import torch
import numpy as np
from groot_rlt.models import RLTActor, RLTCritic
from groot_rlt.training.sac import SAC
from groot_rlt.training.replay_buffer import ReplayBuffer

D_RLT = 64
ACTION_DIM = 7
CHUNK_SIZE = 8
BATCH_SIZE = 32


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def actor(device):
    return RLTActor(d_rlt=D_RLT, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
                    hidden_dim=128, num_layers=2).to(device)


@pytest.fixture
def critic(device):
    return RLTCritic(d_rlt=D_RLT, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
                     hidden_dim=128, num_layers=2).to(device)


@pytest.fixture
def sac(actor, critic, device):
    return SAC(actor=actor, critic=critic, device=device,
               actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3)


@pytest.fixture
def buffer():
    buf = ReplayBuffer(capacity=1000, d_rlt=D_RLT, action_dim=ACTION_DIM,
                       chunk_size=CHUNK_SIZE, human_demo_ratio=0.0, device="cpu")
    # Fill with random transitions
    for _ in range(500):
        buf.add(
            rl_token=np.random.randn(D_RLT).astype(np.float32),
            vla_action=np.random.randn(CHUNK_SIZE, ACTION_DIM).astype(np.float32),
            delta_action=np.random.randn(CHUNK_SIZE, ACTION_DIM).astype(np.float32) * 0.05,
            reward=float(np.random.rand()),
            next_rl_token=np.random.randn(D_RLT).astype(np.float32),
            done=bool(np.random.rand() < 0.1),
        )
    return buf


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=100, d_rlt=D_RLT, action_dim=ACTION_DIM,
                           chunk_size=CHUNK_SIZE, human_demo_ratio=0.0)
        assert len(buf) == 0
        buf.add(
            np.zeros(D_RLT), np.zeros((CHUNK_SIZE, ACTION_DIM)),
            np.zeros((CHUNK_SIZE, ACTION_DIM)), 0.5,
            np.zeros(D_RLT), False,
        )
        assert len(buf) == 1

    def test_circular_overflow(self):
        capacity = 10
        buf = ReplayBuffer(capacity=capacity, d_rlt=D_RLT, action_dim=ACTION_DIM,
                           chunk_size=CHUNK_SIZE, human_demo_ratio=0.0)
        for i in range(20):
            buf.add(np.ones(D_RLT) * i, np.zeros((CHUNK_SIZE, ACTION_DIM)),
                    np.zeros((CHUNK_SIZE, ACTION_DIM)), float(i), np.zeros(D_RLT), False)
        assert len(buf) == capacity  # Capped at capacity

    def test_sample_shapes(self, buffer):
        batch = buffer.sample(BATCH_SIZE)
        assert batch["rl_tokens"].shape == (BATCH_SIZE, D_RLT)
        assert batch["vla_actions"].shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
        assert batch["rewards"].shape == (BATCH_SIZE,)
        assert batch["dones"].shape == (BATCH_SIZE,)

    def test_human_demo_injection(self):
        buf = ReplayBuffer(capacity=500, d_rlt=D_RLT, action_dim=ACTION_DIM,
                           chunk_size=CHUNK_SIZE, human_demo_ratio=0.5)
        # Add RL transitions
        for _ in range(200):
            buf.add(np.zeros(D_RLT), np.zeros((CHUNK_SIZE, ACTION_DIM)),
                    np.zeros((CHUNK_SIZE, ACTION_DIM)), 0.0, np.zeros(D_RLT), False)
        # Add human demos
        for _ in range(10):
            buf.add(np.ones(D_RLT), np.ones((CHUNK_SIZE, ACTION_DIM)),
                    np.zeros((CHUNK_SIZE, ACTION_DIM)), 1.0, np.ones(D_RLT), True,
                    is_human_demo=True)

        assert buf.num_human_demos == 10
        batch = buf.sample(BATCH_SIZE)
        # ~50% should be human demos (reward ≈ 1.0)
        assert batch["rewards"].shape == (BATCH_SIZE,)


class TestSAC:
    def test_single_update(self, sac, buffer, device):
        batch = buffer.sample(BATCH_SIZE)
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        losses = sac.update(batch)

        assert "critic_loss" in losses
        assert "actor_loss" in losses
        assert "alpha_loss" in losses
        assert "alpha" in losses
        # All losses should be finite
        for k, v in losses.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_alpha_is_positive(self, sac, buffer, device):
        batch = {k: v.to(device) for k, v in buffer.sample(BATCH_SIZE).items()}
        for _ in range(5):
            sac.update(batch)
        assert sac.alpha.item() > 0

    def test_target_critic_updates(self, sac, buffer, device):
        """Target critic should change after updates (soft update)."""
        # Get initial target weights
        initial_target = [p.clone() for p in sac.critic_target.parameters()]
        batch = {k: v.to(device) for k, v in buffer.sample(BATCH_SIZE).items()}

        for _ in range(3):
            sac.update(batch)

        # Target should have changed
        for init, current in zip(initial_target, sac.critic_target.parameters()):
            if init.numel() > 0:
                assert not torch.allclose(init, current), "Target critic not updating"
                break

    def test_total_updates_counter(self, sac, buffer, device):
        assert sac.total_updates == 0
        batch = {k: v.to(device) for k, v in buffer.sample(BATCH_SIZE).items()}
        sac.update(batch)
        assert sac.total_updates == 1

    def test_state_dict_roundtrip(self, sac, buffer, device):
        """Save and load state dict — losses should be same after reload."""
        batch = {k: v.to(device) for k, v in buffer.sample(BATCH_SIZE).items()}
        sac.update(batch)

        state = sac.state_dict()
        losses_before = sac.update(batch)

        sac.load_state_dict(state)
        losses_after = sac.update(batch)

        assert abs(losses_before["critic_loss"] - losses_after["critic_loss"]) < 1e-4
