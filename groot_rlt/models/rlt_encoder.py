"""
RLT Encoder-Decoder: Bottleneck transformer that compresses GR00T's VLM hidden
states into a compact RL token. The information bottleneck forces the token to
retain only task-relevant state, which the small actor-critic can efficiently
learn over.

Architecture:
    hidden_states (seq_len, d_model) → Encoder → rl_token (d_rlt,) → Decoder → hidden_states_hat
    Pre-training loss: MSE(hidden_states_hat, hidden_states)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RLTEncoder(nn.Module):
    """
    Transformer encoder that maps VLM hidden states to a single RL token.

    Takes the concatenated hidden states from the last `num_layers` of GR00T's VLM
    (shape: [batch, seq_len * num_layers, d_model]) and compresses them through
    cross-attention pooling into a single vector (the RL token).
    """

    def __init__(
        self,
        d_model: int = 2048,       # GR00T VLM hidden dim (Cosmos-Reason-2B)
        d_rlt: int = 256,           # RL token dimensionality
        num_heads: int = 8,
        num_layers: int = 2,        # Encoder transformer depth
        dropout: float = 0.1,
        num_vlm_layers_to_use: int = 4,  # How many VLM layers to hook
    ):
        super().__init__()
        self.d_model = d_model
        self.d_rlt = d_rlt
        self.num_vlm_layers_to_use = num_vlm_layers_to_use

        # Project from VLM hidden dim to encoder working dim
        self.input_proj = nn.Linear(d_model, d_rlt * 4)

        # Learnable query token for pooling — this becomes the RL token
        self.query_token = nn.Parameter(torch.randn(1, 1, d_rlt * 4))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_rlt * 4,
            nhead=num_heads,
            dim_feedforward=d_rlt * 8,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention pooling: query token attends to all sequence positions
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_rlt * 4,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Project down to RL token dim
        self.output_proj = nn.Sequential(
            nn.Linear(d_rlt * 4, d_rlt * 2),
            nn.GELU(),
            nn.Linear(d_rlt * 2, d_rlt),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model] — VLM hidden states
                           (already concatenated across layers by GR00TWrapperWithHooks)

        Returns:
            rl_token: [batch, d_rlt]
        """
        batch = hidden_states.shape[0]

        # Project to encoder working dim
        x = self.input_proj(hidden_states)  # [B, S, d_rlt*4]

        # Run transformer over sequence
        x = self.transformer(x)  # [B, S, d_rlt*4]

        # Cross-attention pool: single query attends to full sequence
        query = self.query_token.expand(batch, -1, -1)  # [B, 1, d_rlt*4]
        pooled, _ = self.pool_attn(query, x, x)         # [B, 1, d_rlt*4]
        pooled = pooled.squeeze(1)                        # [B, d_rlt*4]

        # Project down to RL token
        rl_token = self.output_proj(pooled)  # [B, d_rlt]
        return rl_token


class RLTDecoder(nn.Module):
    """
    Transformer decoder that reconstructs VLM hidden states from the RL token.
    Used only during offline pre-training (Phase 1) to enforce the information
    bottleneck. Frozen / discarded during online RL (Phase 2).

    Reconstruction loss: MSE(decoder_output, original_hidden_states)
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_rlt: int = 256,
        seq_len: int = 512,        # Expected sequence length of VLM hidden states
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_rlt = d_rlt
        self.seq_len = seq_len

        # Project RL token up to working dim
        self.token_proj = nn.Sequential(
            nn.Linear(d_rlt, d_rlt * 4),
            nn.GELU(),
            nn.Linear(d_rlt * 4, d_rlt * 4),
        )

        # Learnable positional queries (one per sequence position to reconstruct)
        self.pos_queries = nn.Parameter(torch.randn(1, seq_len, d_rlt * 4))

        # Cross-attention: positional queries attend to expanded RL token
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_rlt * 4,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_rlt * 4,
            nhead=num_heads,
            dim_feedforward=d_rlt * 8,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Project back to VLM hidden dim
        self.output_proj = nn.Linear(d_rlt * 4, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, rl_token: torch.Tensor, target_seq_len: int | None = None) -> torch.Tensor:
        """
        Args:
            rl_token: [batch, d_rlt]
            target_seq_len: if provided, use this seq length (default: self.seq_len)

        Returns:
            reconstructed_hidden_states: [batch, seq_len, d_model]
        """
        batch = rl_token.shape[0]
        seq_len = target_seq_len or self.seq_len

        # Expand RL token to key/value for cross-attention
        token_kv = self.token_proj(rl_token).unsqueeze(1)  # [B, 1, d_rlt*4]

        # Positional queries (one per output position)
        queries = self.pos_queries[:, :seq_len, :].expand(batch, -1, -1)  # [B, S, d_rlt*4]

        # Each position attends to the RL token
        x, _ = self.cross_attn(queries, token_kv, token_kv)  # [B, S, d_rlt*4]

        # Refine with self-attention
        x = self.transformer(x)  # [B, S, d_rlt*4]

        # Project back to VLM dim
        reconstructed = self.output_proj(x)  # [B, S, d_model]
        return reconstructed


class RLTBottleneck(nn.Module):
    """
    Combined encoder-decoder for Phase 1 pre-training.
    Trains the encoder to produce RL tokens that contain enough information
    to reconstruct the original VLM hidden states.
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_rlt: int = 256,
        seq_len: int = 512,
        num_heads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        num_vlm_layers_to_use: int = 4,
    ):
        super().__init__()
        self.encoder = RLTEncoder(
            d_model=d_model,
            d_rlt=d_rlt,
            num_heads=num_heads,
            num_layers=encoder_layers,
            dropout=dropout,
            num_vlm_layers_to_use=num_vlm_layers_to_use,
        )
        self.decoder = RLTDecoder(
            d_model=d_model,
            d_rlt=d_rlt,
            seq_len=seq_len,
            num_heads=num_heads,
            num_layers=decoder_layers,
            dropout=dropout,
        )

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            dict with 'rl_token', 'reconstructed', 'loss'
        """
        rl_token = self.encoder(hidden_states)
        reconstructed = self.decoder(rl_token, target_seq_len=hidden_states.shape[1])
        loss = F.mse_loss(reconstructed, hidden_states)
        return {
            "rl_token": rl_token,
            "reconstructed": reconstructed,
            "loss": loss,
        }

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to RL token. Used during online RL."""
        return self.encoder(hidden_states)
