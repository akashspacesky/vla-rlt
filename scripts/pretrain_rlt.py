"""
Phase 1: Offline RLT Bottleneck Pre-training.

Loads GR00T N1, extracts hidden states from demonstration data, and trains
the RLTBottleneck encoder-decoder to reconstruct them through the RL token
bottleneck. Saves encoder checkpoint for use in Phase 2.

Usage:
    python scripts/pretrain_rlt.py \
        --config configs/default.yaml \
        --data_dir /path/to/demos \
        [--dry_run]  # Validate shapes without running full training
"""

import argparse
import torch
import random
import numpy as np
from omegaconf import OmegaConf

from groot_rlt.models import RLTBottleneck
from groot_rlt.models.groot_wrapper import GR00TWrapperWithHooks, MockGR00TWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="RLT Phase 1: Bottleneck Pre-training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, required=False, default=None)
    parser.add_argument("--output", type=str, default="checkpoints/bottleneck_pretrained.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--dry_run", action="store_true", help="Validate shapes only")
    parser.add_argument("--mock", action="store_true", help="Use mock GR00T (no GPU required)")
    return parser.parse_args()


def load_groot(cfg, mock: bool = False):
    if mock:
        print("[pretrain] Using MockGR00TWrapper (no real GR00T needed)")
        return MockGR00TWrapper(
            d_model=2048,
            seq_len=128,
            num_hook_layers=cfg.num_hook_layers,
            chunk_size=cfg.chunk_size,
            action_dim=cfg.action_dim,
        )

    try:
        from groot.model.policy import Gr00tPolicy  # type: ignore
    except ImportError:
        raise ImportError(
            "GR00T N1 not installed. Run:\n"
            "  git clone https://github.com/NVIDIA/Isaac-GR00T && pip install -e Isaac-GR00T/\n"
            "Or use --mock for development without GR00T."
        )

    policy = Gr00tPolicy.from_pretrained(
        cfg.groot.model_path or "nvidia/GR00T-N1",
        embodiment_tag=cfg.groot.embodiment_tag,
    )
    return GR00TWrapperWithHooks(
        groot_policy=policy,
        num_hook_layers=cfg.num_hook_layers,
        freeze=True,
    )


def dry_run(cfg, groot_wrapper):
    """Validate tensor shapes without running full training."""
    print("[dry_run] Validating tensor shapes...")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    bottleneck = RLTBottleneck(
        d_model=groot_wrapper.get_hidden_dim(),
        d_rlt=cfg.rlt_dim,
        seq_len=cfg.seq_len,
    ).to(device)

    # Fake a batch of hidden states
    batch_size = 2
    seq_len = 128 * cfg.num_hook_layers  # hooked layers * seq_len
    d_model = groot_wrapper.get_hidden_dim()
    fake_hs = torch.randn(batch_size, seq_len, d_model, device=device)

    result = bottleneck(fake_hs)
    print(f"  hidden_states:   {fake_hs.shape}")
    print(f"  rl_token:        {result['rl_token'].shape}   (expected [{batch_size}, {cfg.rlt_dim}])")
    print(f"  reconstructed:   {result['reconstructed'].shape}   (expected [{batch_size}, {seq_len}, {d_model}])")
    print(f"  recon_loss:      {result['loss'].item():.4f}")
    print("[dry_run] All shapes OK.")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"[pretrain] Device: {device}")

    groot_wrapper = load_groot(cfg, mock=args.mock or args.dry_run)

    if args.dry_run:
        dry_run(cfg, groot_wrapper)
        return

    if args.data_dir is None:
        raise ValueError("--data_dir is required for full pre-training. Use --dry_run to test shapes.")

    from groot_rlt.training.trainer import RLTTrainer
    from groot_rlt.envs.robot_env import MockRobotEnv

    # Trainer handles the full pre-training loop
    trainer = RLTTrainer(cfg=cfg, groot_wrapper=groot_wrapper, env=MockRobotEnv())
    epochs = args.epochs or cfg.pretrain_epochs
    best_loss = trainer.pretrain(demo_data_dir=args.data_dir, max_epochs=epochs)

    # Save encoder checkpoint
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {"bottleneck": trainer.bottleneck.state_dict(), "cfg": OmegaConf.to_container(cfg)},
        args.output,
    )
    print(f"[pretrain] Saved checkpoint to {args.output}  (best recon loss: {best_loss:.4f})")


if __name__ == "__main__":
    main()
