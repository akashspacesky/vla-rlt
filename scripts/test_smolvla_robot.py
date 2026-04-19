"""
SmolVLA inference + smooth execution test on real SO-101.

Speed optimisations:
  1. --flow-steps N  : reduce SmolVLA flow-matching denoising steps (default 5, was 10)
  2. Pipelined inference: next VLA call runs in a background thread while
     the robot executes the current motion, hiding most of the latency.

Usage:
    # Observe only (safe — no movement):
    python3 scripts/test_smolvla_robot.py --port /dev/tty.usbmodem5A680100951

    # Smooth + fast execution:
    python3 scripts/test_smolvla_robot.py --port /dev/tty.usbmodem5A680100951 \\
        --execute --smooth-steps 20 --step-delay 0.05 --flow-steps 5

    # Custom task:
    python3 scripts/test_smolvla_robot.py --port /dev/tty.usbmodem5A680100951 \\
        --execute --task "pick up the red block" --steps 10
"""

import argparse
import time
import threading
import numpy as np

from rlt.models.smolvla_wrapper import load_smolvla
from rlt.envs.robot_env import SO101Env, SO101_MOTORS
from rlt.utils.device import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="SO-101 USB port")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--task", default="pick up the object and place it in the target")
    p.add_argument("--steps", type=int, default=10, help="Number of VLA inference steps")
    p.add_argument("--model", default="lerobot/smolvla_base")
    p.add_argument("--device", default="auto")
    p.add_argument("--execute", action="store_true",
                   help="Send actions to robot. It WILL MOVE — workspace must be clear.")
    p.add_argument("--smooth-steps", type=int, default=20,
                   help="Interpolation waypoints per action (default 20)")
    p.add_argument("--step-delay", type=float, default=0.05,
                   help="Seconds between waypoints (default 0.05 = 20 Hz)")
    p.add_argument("--flow-steps", type=int, default=5,
                   help="SmolVLA flow-matching denoising steps (default 5, max 10)")
    p.add_argument("--calibrate", action="store_true",
                   help="Run interactive motor calibration (once per setup).")
    return p.parse_args()


def set_flow_steps(policy, n: int):
    """Override SmolVLA flow-matching step count at runtime."""
    policy.model.config.num_steps = n


def smooth_move(env: SO101Env, current_deg: np.ndarray, target_deg: np.ndarray,
                n_steps: int, step_delay: float):
    """Linearly interpolate from current to target joint positions."""
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        waypoint = current_deg + alpha * (target_deg - current_deg)
        action_dict = {f"{m}.pos": float(waypoint[j]) for j, m in enumerate(SO101_MOTORS)}
        env._robot.send_action(action_dict)
        time.sleep(step_delay)


def infer(vla, obs: dict) -> dict:
    return vla(obs)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"[*] Device: {device}")

    print(f"[*] Loading SmolVLA ({args.model}) ...")
    vla = load_smolvla(model_id=args.model, device=device)
    set_flow_steps(vla.policy, args.flow_steps)
    print(f"[*] SmolVLA ready. flow_steps={args.flow_steps}, "
          f"hidden_dim={vla.get_hidden_dim()}, chunk={vla.get_chunk_size()}")

    env = SO101Env(
        task_description=args.task,
        port=args.port,
        camera_name="top",
        image_size=224,
        fps=30,
        calibrate=args.calibrate,
    )

    if args.execute:
        motion_s = args.smooth_steps * args.step_delay
        print(f"\n  *** --execute ON | smooth: {args.smooth_steps}×{args.step_delay}s={motion_s:.1f}s "
              f"| flow_steps={args.flow_steps} ***")
        print("  Ctrl+C aborts at any time. Pausing 3s...\n")
        time.sleep(3)

    try:
        print("[*] Connecting hardware ...")
        obs = env.reset()
        print("[*] Connected.\n")

        # first inference (no pipeline on step 1)
        t0 = time.perf_counter()
        out = infer(vla, obs)
        print(f"[warmup] first inference: {(time.perf_counter()-t0)*1000:.0f}ms")

        for step in range(1, args.steps + 1):
            target  = out["action_chunk"][0, 0].cpu().numpy()
            current = obs["observation.state"][0].cpu().numpy()
            hidden  = out["hidden_states"]

            print(f"[step {step}/{args.steps}]")
            print(f"  current (deg) : {np.array2string(current, precision=1, suppress_small=True)}")
            print(f"  target  (deg) : {np.array2string(target,  precision=1, suppress_small=True)}")
            print(f"  delta   (deg) : {np.array2string(target - current, precision=1, suppress_small=True)}")
            print(f"  hidden_states : {tuple(hidden.shape)}")

            if args.execute:
                # Run motion in background; run next inference concurrently
                next_obs_container: list = []

                def pipeline_step():
                    smooth_move(env, current, target, args.smooth_steps, args.step_delay)
                    next_obs_container.append(env.reset())

                motion_thread = threading.Thread(target=pipeline_step, daemon=True)
                motion_thread.start()

                t_inf = time.perf_counter()
                next_out = infer(vla, obs)  # infer on current obs while robot moves
                inf_ms = (time.perf_counter() - t_inf) * 1000

                motion_thread.join()

                obs = next_obs_container[0]
                out = next_out
                print(f"  [DONE] inference={inf_ms:.0f}ms (pipelined with "
                      f"{args.smooth_steps * args.step_delay:.1f}s motion)")
            else:
                t_inf = time.perf_counter()
                obs = env.reset()
                out = infer(vla, obs)
                print(f"  inference: {(time.perf_counter()-t_inf)*1000:.0f}ms")

            print()

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        env.close()
        print("[*] Disconnected cleanly.")


if __name__ == "__main__":
    main()
