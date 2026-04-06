"""
Evaluate a trained RLT policy.

Runs N episodes and reports: success rate, mean reward, mean episode length.
Compares against VLA baseline (delta=0) if --compare_baseline is set.

Usage:
    python scripts/evaluate.py \
        --config configs/tasks/ethernet_insertion.yaml \
        --checkpoint runs/rlt_ethernet_insertion/sac_latest.pt \
        --encoder_ckpt checkpoints/bottleneck_pretrained.pt \
        --num_episodes 50 \
        [--deterministic] \
        [--compare_baseline]
"""

import argparse
import torch
import numpy as np
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RLT policy")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--deterministic", action="store_true", help="Use mean action (no sampling)")
    parser.add_argument("--compare_baseline", action="store_true", help="Also evaluate VLA-only baseline")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def run_episodes(actor, bottleneck, groot_wrapper, env, num_episodes, deterministic, device):
    """Run N episodes and return metrics."""
    successes = []
    rewards = []
    lengths = []

    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done:
            groot_out = groot_wrapper(obs)
            with torch.no_grad():
                rl_token = bottleneck.encode(
                    groot_out["hidden_states"].unsqueeze(0).to(device)
                )
                vla_action = groot_out["action_chunk"].to(device)
                action = actor.get_action(rl_token, vla_action.unsqueeze(0), deterministic=deterministic)
                action = action.squeeze(0).cpu().numpy()

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

        successes.append(float(info.get("success", False)))
        rewards.append(ep_reward)
        lengths.append(ep_steps)
        print(f"  Episode {ep+1:3d}/{num_episodes}  success={successes[-1]:.0f}  reward={ep_reward:.2f}")

    return {
        "success_rate": np.mean(successes),
        "mean_reward": np.mean(rewards),
        "mean_length": np.mean(lengths),
        "std_reward": np.std(rewards),
    }


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    if args.mock:
        from groot_rlt.models.groot_wrapper import MockGR00TWrapper
        from groot_rlt.envs.robot_env import MockRobotEnv
        groot_wrapper = MockGR00TWrapper(
            d_model=2048, seq_len=128, num_hook_layers=cfg.num_hook_layers,
            chunk_size=cfg.chunk_size, action_dim=cfg.action_dim,
        )
        env = MockRobotEnv(chunk_size=cfg.chunk_size, action_dim=cfg.action_dim)
    else:
        raise NotImplementedError("Connect real robot env. Use --mock for testing.")

    from groot_rlt.models import RLTBottleneck, RLTActor, RLTCritic

    bottleneck = RLTBottleneck(
        d_model=groot_wrapper.get_hidden_dim(),
        d_rlt=cfg.rlt_dim,
        seq_len=cfg.seq_len,
    ).to(device)

    actor = RLTActor(
        d_rlt=cfg.rlt_dim,
        action_dim=cfg.action_dim,
        chunk_size=cfg.chunk_size,
        hidden_dim=cfg.actor.hidden_dim,
        num_layers=cfg.actor.num_layers,
    ).to(device)

    if args.encoder_ckpt:
        enc_state = torch.load(args.encoder_ckpt, map_location=device)
        bottleneck.load_state_dict(enc_state.get("bottleneck", enc_state))

    ckpt = torch.load(args.checkpoint, map_location=device)
    actor.load_state_dict(ckpt["sac"]["actor"])

    actor.eval()
    bottleneck.eval()

    print(f"\n[eval] RLT Policy — {args.num_episodes} episodes, deterministic={args.deterministic}")
    rlt_metrics = run_episodes(actor, bottleneck, groot_wrapper, env, args.num_episodes, args.deterministic, device)

    print(f"\n{'─'*50}")
    print(f"  RLT success rate:  {rlt_metrics['success_rate']*100:.1f}%")
    print(f"  Mean reward:       {rlt_metrics['mean_reward']:.3f} ± {rlt_metrics['std_reward']:.3f}")
    print(f"  Mean ep length:    {rlt_metrics['mean_length']:.1f} steps")

    if args.compare_baseline:
        print(f"\n[eval] VLA Baseline (delta=0) — {args.num_episodes} episodes")

        class ZeroDeltaActor:
            def get_action(self, rl_token, vla_action, deterministic=False):
                return vla_action  # No edit

        baseline_metrics = run_episodes(
            ZeroDeltaActor(), bottleneck, groot_wrapper, env,
            args.num_episodes, deterministic=True, device=device,
        )
        print(f"\n  Baseline success rate:  {baseline_metrics['success_rate']*100:.1f}%")
        improvement = rlt_metrics['success_rate'] - baseline_metrics['success_rate']
        print(f"  RLT improvement:        {improvement*100:+.1f}%")

    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
