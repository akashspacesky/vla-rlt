"""
Phase 2: Online SAC Training on the SO-101.

Usage:
    # Mock (no hardware):
    python scripts/train.py --mock

    # Real SO-101:
    python scripts/train.py --config configs/tasks/so101_pick_place.yaml \\
        --encoder_ckpt checkpoints/bottleneck_pretrained.pt
"""

import argparse
import torch
import random
import numpy as np
from omegaconf import OmegaConf

from rlt.utils.device import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--encoder_ckpt", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--port", default="/dev/ttyUSB0", help="SO-101 USB serial port")
    p.add_argument("--cam", type=int, default=0, help="USB camera index")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = get_device(cfg.device)
    cfg.device = device
    print(f"[train] Device: {device}")

    # Load VLA
    if args.mock:
        from rlt.models.smolvla_wrapper import MockVLAWrapper
        vla = MockVLAWrapper(num_hook_layers=cfg.vla.num_hook_layers,
                             chunk_size=cfg.chunk_size, action_dim=cfg.action_dim)
    else:
        from rlt.models.smolvla_wrapper import load_smolvla
        vla = load_smolvla(
            model_id=cfg.vla.model_id,
            device=device,
            num_hook_layers=cfg.vla.num_hook_layers,
            chunk_size=cfg.chunk_size,
            action_dim=cfg.action_dim,
        )

    # Load env
    if args.mock:
        from rlt.envs.robot_env import MockRobotEnv
        env = MockRobotEnv(chunk_size=cfg.chunk_size, action_dim=cfg.action_dim)
    else:
        from rlt.envs.robot_env import SO101Env
        env = SO101Env(chunk_size=cfg.chunk_size, port=args.port, camera_name="top")

    from rlt.training.trainer import RLTTrainer
    trainer = RLTTrainer(cfg=cfg, vla=vla, env=env)

    if args.encoder_ckpt:
        trainer.load_pretrained_encoder(args.encoder_ckpt)
    else:
        print("[train] WARNING: No encoder checkpoint — encoder weights are random.")

    if args.resume:
        trainer.load_sac(args.resume)

    trainer.train(total_steps=args.steps or cfg.total_steps)
    trainer.logger.finish()


if __name__ == "__main__":
    main()
