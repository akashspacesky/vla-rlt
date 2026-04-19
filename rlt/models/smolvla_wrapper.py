"""
SmolVLA Wrapper with hidden-state hooks.

SmolVLA (HuggingFace LeRobot) is a 450M VLA built on SmolVLM-2.
- VLM backbone: SmolVLM-2 (hidden_size=960, 32 layers)
- Action expert: lightweight transformer (~100M), interleaved cross/self-attn
- Visual tokens: 64 per frame (PixelShuffle 512x512 → 64 tokens)
- Trained on SO-100/SO-101 data

Layer hook note:
  SmolVLMWithExpertModel.forward() applies VLM layers manually (never calling
  layer.forward()), so hooks on transformer layers won't fire. We hook
  layer.mlp instead, which IS called via layer.mlp(hidden_states) and goes
  through its own .forward(). Output shape: [B, seq_len_prefix, hidden_size].

MPS note: SmolVLA runs on Apple Silicon via device="mps".
          Keep batch_size=1 during inference for 8GB machines.

Install LeRobot first:
    git clone https://github.com/huggingface/lerobot
    uv pip install -e "lerobot/[smolvla]"
"""

import torch
import torch.nn as nn
from typing import Any

from rlt.models.vla_backend import VLABackend


# SmolVLM-2 256M config (backbone of SmolVLA)
SMOLVLM_HIDDEN_SIZE = 960
SMOLVLM_NUM_LAYERS = 32
SMOLVLA_CHUNK_SIZE = 16
SMOLVLA_ACTION_DIM = 6      # SO-101: 6 DoF (5 joints + gripper)

# LeRobot constants
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


class HiddenStateCapture:
    """Captures output tensors from registered modules via forward hooks."""

    def __init__(self):
        self.states: list[torch.Tensor] = []
        self._hooks: list[Any] = []

    def register(self, module: nn.Module):
        self._hooks.append(module.register_forward_hook(self._hook_fn))

    def _hook_fn(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        self.states.append(h.detach())

    def clear(self):
        self.states.clear()

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class SmolVLAWrapper(VLABackend):
    """
    Wraps SmolVLAPolicy from LeRobot to expose VLM hidden states.

    Hooks the MLP sub-module of the last `num_hook_layers` VLM transformer
    layers. The MLP is called via layer.mlp(hidden_states) during the
    SmolVLMWithExpertModel forward pass, so hooks fire correctly.

    Args:
        policy: Instantiated SmolVLAPolicy (from lerobot)
        num_hook_layers: How many VLM layers to hook (default 4)
        chunk_size: Action chunk length (must match policy config)
        action_dim: Per-step action dim (6 for SO-101)
        freeze: Freeze all SmolVLA params during RL (default True)
        device: "mps", "cuda", or "cpu"
        max_lang_len: Max token length for language tokenization (default 64)
    """

    def __init__(
        self,
        policy: nn.Module,
        num_hook_layers: int = 4,
        chunk_size: int = SMOLVLA_CHUNK_SIZE,
        action_dim: int = SMOLVLA_ACTION_DIM,
        freeze: bool = True,
        device: str = "mps",
        max_lang_len: int = 64,
    ):
        super().__init__()
        self.policy = policy.to(device)
        self.num_hook_layers = num_hook_layers
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self.device = device
        self.max_lang_len = max_lang_len
        self.capture = HiddenStateCapture()

        if freeze:
            for p in self.policy.parameters():
                p.requires_grad_(False)

        self._register_hooks()

    def _get_vlm_layers(self) -> list[nn.Module]:
        """
        Navigate SmolVLAPolicy → SmolVLMWithExpertModel → VLM transformer layers.

        Path:
          policy                              SmolVLAPolicy
          policy.model                        VLAFlowMatching
          policy.model.vlm_with_expert        SmolVLMWithExpertModel
          policy.model.vlm_with_expert.vlm    SmolVLMForConditionalGeneration
          .vlm.model.text_model.layers        nn.ModuleList of transformer layers
        """
        try:
            layers = self.policy.model.vlm_with_expert.vlm.model.text_model.layers
            if hasattr(layers, "__len__") and len(layers) > 0:
                return list(layers)
        except AttributeError:
            pass

        raise RuntimeError(
            "Could not locate SmolVLM text_model.layers.\n"
            "Expected: policy.model.vlm_with_expert.vlm.model.text_model.layers\n"
            "Run: print([n for n, _ in policy.named_modules()]) to inspect."
        )

    def _register_hooks(self):
        """
        Hook layer.mlp (not the layer itself) because SmolVLMWithExpertModel
        calls layer components manually — layer.forward() is never invoked.
        layer.mlp(hidden) IS called normally, so its forward hook fires.
        """
        layers = self._get_vlm_layers()
        for layer in layers[-self.num_hook_layers:]:
            self.capture.register(layer.mlp)

    def _build_batch(self, obs: dict) -> dict:
        """
        Convert raw obs dict into the batch format SmolVLAPolicy expects.

        Auto-remaps obs image keys to the checkpoint's expected camera keys
        and resizes to the checkpoint's required resolution. This lets a
        single-camera env work against multi-camera base checkpoints.
        """
        import torch.nn.functional as F

        batch = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, dtype=torch.float32)
            else:
                batch[k] = v

        # Remap obs image keys to whatever keys the loaded checkpoint expects,
        # resizing to the checkpoint's required resolution.
        expected_img_features = dict(self.policy.config.image_features)
        obs_img_keys = [k for k in obs if k.startswith("observation.images.")]
        if obs_img_keys and expected_img_features:
            src_imgs = [batch[k] for k in obs_img_keys if k in batch]
            for i, (exp_key, feat) in enumerate(expected_img_features.items()):
                if exp_key not in batch:
                    img = src_imgs[i % len(src_imgs)]
                    tgt_h, tgt_w = feat.shape[1], feat.shape[2]
                    if img.shape[-2] != tgt_h or img.shape[-1] != tgt_w:
                        img = F.interpolate(img, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
                    batch[exp_key] = img

        # Tokenize task language
        if "task" in obs:
            texts = obs["task"] if isinstance(obs["task"], list) else [obs["task"]]
            processor = self.policy.model.vlm_with_expert.processor
            tokens = processor.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_lang_len,
            )
            batch[OBS_LANGUAGE_TOKENS] = tokens["input_ids"].to(self.device)
            batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].to(self.device)

        return batch

    @torch.no_grad()
    def forward(self, obs: dict) -> dict[str, torch.Tensor]:
        """
        Args:
            obs: {
                "observation.images.<cam>": [B, C, H, W] float32 in [0,1]
                "observation.state":        [B, state_dim]
                "task":                     list[str]
            }

        Returns:
            {
                "action_chunk":  [B, chunk_size, action_dim]
                "hidden_states": [B, seq_len * num_hook_layers, d_model]
            }
        """
        self.capture.clear()
        batch = self._build_batch(obs)
        action_chunk = self.policy.predict_action_chunk(batch)

        if not self.capture.states:
            raise RuntimeError(
                "No hidden states captured after forward pass.\n"
                "Verify that SmolVLMWithExpertModel.forward() is being called "
                "and that layer.mlp hooks are still registered.\n"
                "Run: print([n for n, _ in policy.named_modules()])"
            )

        hidden_states = torch.cat(self.capture.states, dim=1)
        return {
            "action_chunk": action_chunk,
            "hidden_states": hidden_states,
        }

    def get_hidden_dim(self) -> int:
        try:
            return self.policy.model.vlm_with_expert.vlm.model.text_model.config.hidden_size
        except AttributeError:
            return SMOLVLM_HIDDEN_SIZE

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def get_action_dim(self) -> int:
        return self._action_dim


def load_smolvla(
    model_id: str = "lerobot/smolvla_base",
    device: str = "mps",
    num_hook_layers: int = 4,
    chunk_size: int = SMOLVLA_CHUNK_SIZE,
    action_dim: int = SMOLVLA_ACTION_DIM,
) -> SmolVLAWrapper:
    """
    Load SmolVLA from HuggingFace Hub and wrap it.

    Usage:
        wrapper = load_smolvla(device="mps")
        out = wrapper(obs)
    """
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        raise ImportError(
            "LeRobot not installed. Run:\n"
            "  git clone https://github.com/huggingface/lerobot\n"
            "  uv pip install -e 'lerobot/[smolvla]'"
        )

    print(f"[SmolVLA] Loading {model_id} \u2192 {device} ...")
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.eval()

    return SmolVLAWrapper(
        policy=policy,
        num_hook_layers=num_hook_layers,
        chunk_size=chunk_size,
        action_dim=action_dim,
        freeze=True,
        device=device,
    )


class MockVLAWrapper(VLABackend):
    """
    Mock VLA backend for unit testing and dry-runs.
    No model weights needed. Produces correctly-shaped random tensors.
    """

    def __init__(
        self,
        d_model: int = SMOLVLM_HIDDEN_SIZE,
        seq_len: int = 64,
        num_hook_layers: int = 4,
        chunk_size: int = SMOLVLA_CHUNK_SIZE,
        action_dim: int = SMOLVLA_ACTION_DIM,
        device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_hook_layers = num_hook_layers
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self.device = device
        self._dummy = nn.Linear(1, 1)

    @torch.no_grad()
    def forward(self, obs: dict) -> dict[str, torch.Tensor]:
        batch = 1
        for v in obs.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                batch = v.shape[0]
                break
        hidden_states = torch.randn(
            batch, self.seq_len * self.num_hook_layers, self.d_model, device=self.device
        )
        action_chunk = torch.randn(
            batch, self._chunk_size, self._action_dim, device=self.device
        )
        return {"action_chunk": action_chunk, "hidden_states": hidden_states}

    def get_hidden_dim(self) -> int:
        return self.d_model

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def get_action_dim(self) -> int:
        return self._action_dim
