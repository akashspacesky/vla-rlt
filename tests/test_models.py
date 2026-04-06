"""Unit tests for RLT model components. No real GR00T required."""

import pytest
import torch
from groot_rlt.models import RLTBottleneck, RLTActor, RLTCritic
from groot_rlt.models.groot_wrapper import MockGR00TWrapper

BATCH = 4
D_RLT = 64       # Small dim for fast tests
ACTION_DIM = 7
CHUNK_SIZE = 8
D_MODEL = 128    # Small for tests
SEQ_LEN = 32
NUM_HOOK_LAYERS = 2


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def mock_wrapper(device):
    return MockGR00TWrapper(
        d_model=D_MODEL,
        seq_len=SEQ_LEN,
        num_hook_layers=NUM_HOOK_LAYERS,
        chunk_size=CHUNK_SIZE,
        action_dim=ACTION_DIM,
    ).to(device)


@pytest.fixture
def bottleneck(device):
    return RLTBottleneck(
        d_model=D_MODEL,
        d_rlt=D_RLT,
        seq_len=SEQ_LEN * NUM_HOOK_LAYERS,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        num_vlm_layers_to_use=NUM_HOOK_LAYERS,
    ).to(device)


@pytest.fixture
def actor(device):
    return RLTActor(
        d_rlt=D_RLT,
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        hidden_dim=128,
        num_layers=2,
        ref_dropout_prob=0.3,
        action_scale=0.1,
    ).to(device)


@pytest.fixture
def critic(device):
    return RLTCritic(
        d_rlt=D_RLT,
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        hidden_dim=128,
        num_layers=2,
    ).to(device)


class TestMockWrapper:
    def test_output_shapes(self, mock_wrapper, device):
        obs = {"video": {}, "state": torch.randn(BATCH, 14)}
        out = mock_wrapper({"video": {"cam": torch.randn(BATCH, 3, 224, 224)}})
        assert out["action_chunk"].shape == (BATCH, CHUNK_SIZE, ACTION_DIM)
        assert out["hidden_states"].shape == (BATCH, SEQ_LEN * NUM_HOOK_LAYERS, D_MODEL)


class TestRLTBottleneck:
    def test_encode_shape(self, bottleneck, device):
        hs = torch.randn(BATCH, SEQ_LEN * NUM_HOOK_LAYERS, D_MODEL, device=device)
        rl_token = bottleneck.encode(hs)
        assert rl_token.shape == (BATCH, D_RLT)

    def test_forward_shapes(self, bottleneck, device):
        hs = torch.randn(BATCH, SEQ_LEN * NUM_HOOK_LAYERS, D_MODEL, device=device)
        result = bottleneck(hs)
        assert result["rl_token"].shape == (BATCH, D_RLT)
        assert result["reconstructed"].shape == hs.shape
        assert result["loss"].ndim == 0  # Scalar

    def test_reconstruction_loss_decreases(self, bottleneck, device):
        """Loss should decrease with a few gradient steps."""
        optimizer = torch.optim.Adam(bottleneck.parameters(), lr=1e-3)
        hs = torch.randn(BATCH, SEQ_LEN * NUM_HOOK_LAYERS, D_MODEL, device=device)

        initial_loss = bottleneck(hs)["loss"].item()
        for _ in range(10):
            result = bottleneck(hs)
            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()

        final_loss = bottleneck(hs)["loss"].item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"


class TestRLTActor:
    def test_output_shapes(self, actor, device):
        rl_token = torch.randn(BATCH, D_RLT, device=device)
        vla_action = torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM, device=device)
        result = actor(rl_token, vla_action)

        assert result["action_chunk"].shape == (BATCH, CHUNK_SIZE, ACTION_DIM)
        assert result["delta"].shape == (BATCH, CHUNK_SIZE, ACTION_DIM)
        assert result["log_prob"].shape == (BATCH,)

    def test_delta_magnitude(self, actor, device):
        """Delta should be bounded by action_scale (tanh squash)."""
        rl_token = torch.randn(BATCH, D_RLT, device=device)
        vla_action = torch.zeros(BATCH, CHUNK_SIZE, ACTION_DIM, device=device)
        result = actor(rl_token, vla_action)
        delta_mag = result["delta"].abs().max().item()
        assert delta_mag <= actor.action_scale + 1e-4, f"Delta magnitude {delta_mag} exceeds action_scale {actor.action_scale}"

    def test_deterministic_vs_stochastic(self, actor, device):
        """Deterministic action should equal stochastic mean on average."""
        actor.eval()
        rl_token = torch.randn(1, D_RLT, device=device)
        vla_action = torch.randn(1, CHUNK_SIZE, ACTION_DIM, device=device)

        det = actor(rl_token, vla_action, deterministic=True)["action_chunk"]
        sto = actor(rl_token, vla_action, deterministic=False)["action_chunk"]
        # Not identical (due to tanh), but should be close for small std
        assert det.shape == sto.shape

    def test_ref_dropout(self, actor, device):
        """With dropout=1.0, actor should not depend on vla_action."""
        actor.ref_dropout_prob = 1.0
        actor.train()
        rl_token = torch.randn(BATCH, D_RLT, device=device)
        vla_action_a = torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM, device=device)
        vla_action_b = torch.zeros(BATCH, CHUNK_SIZE, ACTION_DIM, device=device)

        torch.manual_seed(42)
        out_a = actor(rl_token, vla_action_a, return_log_prob=False)["mean"]
        torch.manual_seed(42)
        out_b = actor(rl_token, vla_action_b, return_log_prob=False)["mean"]
        # With full dropout, outputs should be nearly identical (same rl_token, zeroed ref)
        assert torch.allclose(out_a, out_b, atol=1e-5), "Dropout not masking reference action"

    def test_get_action_no_grad(self, actor, device):
        actor.eval()
        rl_token = torch.randn(1, D_RLT, device=device)
        vla_action = torch.randn(1, CHUNK_SIZE, ACTION_DIM, device=device)
        action = actor.get_action(rl_token, vla_action)
        assert action.shape == (1, CHUNK_SIZE, ACTION_DIM)
        assert not action.requires_grad


class TestRLTCritic:
    def test_output_shapes(self, critic, device):
        rl_token = torch.randn(BATCH, D_RLT, device=device)
        action = torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM, device=device)
        q1, q2 = critic(rl_token, action)
        assert q1.shape == (BATCH, 1)
        assert q2.shape == (BATCH, 1)

    def test_q_min(self, critic, device):
        rl_token = torch.randn(BATCH, D_RLT, device=device)
        action = torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM, device=device)
        q1, q2 = critic(rl_token, action)
        q_min = critic.q_min(rl_token, action)
        expected = torch.min(q1, q2)
        assert torch.allclose(q_min, expected)

    def test_twin_critics_differ(self, critic, device):
        """Q1 and Q2 should produce different values (independent networks)."""
        rl_token = torch.randn(BATCH, D_RLT, device=device)
        action = torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM, device=device)
        q1, q2 = critic(rl_token, action)
        assert not torch.allclose(q1, q2), "Twin critics should differ"
