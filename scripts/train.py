"""
Phase 2: Online SAC Training.

Loads the pre-trained RLT encoder (frozen), then runs SAC on the robot.
Only actor and critic are updated. GR00T and encoder stay frozen.

Usage:
    python scripts/train.py \
        --config configs/tasks/ethernet_insertion.yaml \
        --encoder_ckpt checkpoints/bottleneck_pretrained.pt \
        [--mock]   # use mock robot + mock GR00T for testing
"""

import argparse
import torch
import random
import numpy as np
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="RLT Phase 2: Online SAC Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--encoder_ckpt", type=str, default=None,
                        help="Path to Phase 1 bottleneck checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to SAC checkpoint to resume from")
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--mock", action="store_true", help="Use mock GR00T + mock robot env")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.device = device
    print(f"[train] Device: {device}")

    # Load GR00T wrapper
    if args.mock:
        from groot_rlt.models.groot_wrapper import MockGR00TWrapper
        groot_wrapper = MockGR00TWrapper(
            d_model=2048,
            seq_len=128,
            num_hook_layers=cfg.num_hook_layers,
            chunk_size=cfg.chunk_size,
            action_dim=cfg.action_dim,
        )
    else:
        from scripts.pretrain_rlt import load_groot
        groot_wrapper = load_groot(cfg)

    # Load environment
    if args.mock:
        from groot_rlt.envs.robot_env import MockRobotEnv
        env = MockRobotEnv(chunk_size=cfg.chunk_size, action_dim=cfg.action_dim)
    else:
        # Real robot: implement your robot env here by subclassing RobotEnv
        raise NotImplementedError(
            "Connect your real robot environment. "
            "Subclass groot_rlt.envs.robot_env.RobotEnv and pass it here.\n"
            "Use --mock for development without hardware."
        )

    from groot_rlt.training.trainer import RLTTrainer

    trainer = RLTTrainer(cfg=cfg, groot_wrapper=groot_wrapper, env=env)

    # Load encoder from Phase 1
    if args.encoder_ckpt:
        trainer.load_pretrained_encoder(args.encoder_ckpt)
    else:
        print("[train] WARNING: No encoder checkpoint provided. Encoder weights are random.")
        print("         Run scripts/pretrain_rlt.py first for best results.")

    # Resume SAC if continuing a run
    if args.resume:
        trainer.load_sac(args.resume)

    total_steps = args.steps or cfg.total_steps
    trainer.train(total_steps=total_steps)

    trainer.logger.finish()
    print("[train] Done.")


if __name__ == "__main__":
    main()
