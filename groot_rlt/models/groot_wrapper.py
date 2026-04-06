"""
GR00T N1 Wrapper with Hidden State Hooks.

GR00T N1 uses a Cosmos-Reason-2B VLM backbone. This wrapper intercepts the
VLM's forward pass using PyTorch hooks to capture internal hidden states from
the last N transformer layers, which are then fed to the RLT encoder.

Usage:
    wrapper = GR00TWrapperWithHooks(groot_model, num_hook_layers=4)
    obs = {"video": ..., "state": ..., "annotation": {"human.validity": [...]}}
    result = wrapper(obs)
    # result["hidden_states"]: [batch, seq_len, d_model] (concatenated hook layers)
    # result["action_chunk"]:  [batch, chunk_size, action_dim]

Note: GR00T must be installed (pip install -e Isaac-GR00T/)
"""

import torch
import torch.nn as nn
from typing import Any


class HiddenStateCapture:
    """Context manager that captures hidden states from transformer layers via hooks."""

    def __init__(self):
        self.hidden_states: list[torch.Tensor] = []
        self._hooks: list[Any] = []

    def register(self, module: nn.Module):
        """Register a forward hook on a transformer layer."""
        hook = module.register_forward_hook(self._hook_fn)
        self._hooks.append(hook)

    def _hook_fn(self, module, input, output):
        # TransformerLayer outputs: (hidden_state, ...) — take first element
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        self.hidden_states.append(h.detach())

    def clear(self):
        self.hidden_states.clear()

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        self.clear()
        return self

    def __exit__(self, *args):
        pass


class GR00TWrapperWithHooks(nn.Module):
    """
    Wraps a GR00T N1 policy to expose VLM internal hidden states alongside
    predicted action chunks.

    The hidden states from the last `num_hook_layers` VLM transformer layers
    are captured and concatenated along the sequence dimension, giving the RLT
    encoder a richer view of the VLM's internal representation.

    Args:
        groot_policy: An instantiated GR00T N1 policy object
                      (groot.model.policy.Gr00tPolicy or similar)
        num_hook_layers: How many of the VLM's last layers to hook (default 4)
        freeze: Whether to freeze all GR00T parameters (default True)
    """

    def __init__(
        self,
        groot_policy: nn.Module,
        num_hook_layers: int = 4,
        freeze: bool = True,
    ):
        super().__init__()
        self.groot_policy = groot_policy
        self.num_hook_layers = num_hook_layers
        self.capture = HiddenStateCapture()

        if freeze:
            for param in self.groot_policy.parameters():
                param.requires_grad_(False)

        self._register_hooks()

    def _get_vlm_layers(self) -> list[nn.Module]:
        """
        Navigate GR00T's model structure to find VLM transformer layers.
        GR00T N1 wraps Cosmos-Reason-2B, which uses a standard LlamaModel-style
        transformer under the hood.

        Tries common attribute paths in order.
        """
        model = self.groot_policy

        # Common attribute paths in GR00T / HuggingFace VLMs
        candidate_paths = [
            ["vlm", "model", "layers"],
            ["model", "vlm", "model", "layers"],
            ["backbone", "model", "layers"],
            ["vlm", "language_model", "model", "layers"],
        ]

        for path in candidate_paths:
            obj = model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if hasattr(obj, "__len__") and len(obj) > 0:
                    return list(obj)
            except AttributeError:
                continue

        raise RuntimeError(
            "Could not find VLM transformer layers in GR00T model. "
            "Check model structure and update _get_vlm_layers() accordingly. "
            f"Available top-level attrs: {list(vars(self.groot_policy).keys())}"
        )

    def _register_hooks(self):
        layers = self._get_vlm_layers()
        # Hook the last num_hook_layers layers
        hook_layers = layers[-self.num_hook_layers:]
        for layer in hook_layers:
            self.capture.register(layer)

    @torch.no_grad()
    def forward(self, obs: dict) -> dict[str, torch.Tensor]:
        """
        Run GR00T forward pass and capture hidden states.

        Args:
            obs: GR00T observation dict with keys:
                 - "video": dict of camera tensors [B, T, C, H, W]
                 - "state": robot state tensor [B, state_dim]
                 - "annotation": dict with language instruction

        Returns:
            dict with:
                - "action_chunk": [B, chunk_size, action_dim]
                - "hidden_states": [B, seq_len * num_hook_layers, d_model]
        """
        self.capture.clear()

        # GR00T forward pass (produces action chunk)
        action_chunk = self.groot_policy.get_action(obs)

        # Concatenate captured hidden states from all hooked layers
        if not self.capture.hidden_states:
            raise RuntimeError(
                "No hidden states captured. Check that hooks are registered correctly."
            )

        # Each captured tensor: [B, seq_len, d_model]
        hidden_states = torch.cat(self.capture.hidden_states, dim=1)  # [B, S*L, d_model]

        return {
            "action_chunk": action_chunk,
            "hidden_states": hidden_states,
        }

    def get_hidden_dim(self) -> int:
        """Returns VLM hidden dim (d_model). Cosmos-Reason-2B uses 2048."""
        # Try to infer from model config
        try:
            return self.groot_policy.vlm.config.hidden_size
        except AttributeError:
            return 2048  # Cosmos-Reason-2B default


class MockGR00TWrapper(nn.Module):
    """
    Mock GR00T wrapper for unit testing and development without the full model.
    Produces random but correctly-shaped tensors.
    """

    def __init__(
        self,
        d_model: int = 2048,
        seq_len: int = 128,
        num_hook_layers: int = 4,
        chunk_size: int = 16,
        action_dim: int = 7,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_hook_layers = num_hook_layers
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Small learned projection to make it trainable in tests
        self.dummy = nn.Linear(1, 1)

    @torch.no_grad()
    def forward(self, obs: dict) -> dict[str, torch.Tensor]:
        batch = 1
        if "video" in obs:
            # Infer batch from first camera tensor
            first_cam = next(iter(obs["video"].values()))
            batch = first_cam.shape[0]

        device = next(self.parameters()).device
        hidden_states = torch.randn(
            batch,
            self.seq_len * self.num_hook_layers,
            self.d_model,
            device=device,
        )
        action_chunk = torch.randn(batch, self.chunk_size, self.action_dim, device=device)
        return {"action_chunk": action_chunk, "hidden_states": hidden_states}

    def get_hidden_dim(self) -> int:
        return self.d_model
