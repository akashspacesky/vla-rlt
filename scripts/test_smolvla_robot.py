"""
SmolVLA inference + smooth execution test on real SO-101.

Loads SmolVLA, connects SO-101 + USB cam, runs N inference steps.
With --execute, interpolates smoothly from current joint state to the
SmolVLA target to avoid sudden jumps.

Usage:
    # Observe only (safe — no movement):
    python3 scripts/test_smolvla_robot.py --port /dev/tty.usbmodem5A680100951

    # Smooth execution, 20 interpolation steps per action:
    python3 scripts/test_smolvla_robot.py --port /dev/tty.usbmodem5A680100951 \\
        --execute --smooth-steps 20 --step-delay 0.05

    # Custom task:
    python3 scripts/test_smolvla_robot.py --port /dev/tty.usbmodem5A680100951 \\
        --execute --task "pick up the red block" --steps 10
"""

import argparse
import time
import numpy as np

from rlt.models.smolvla_wrapper import load_smolvla
from rlt.envs.robot_env import SO101Env, SO101_MOTORS
from rlt.utils.device import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="SO-101 USB port, e.g. /dev/tty.usbmodem...")
    p.add_argument("--cam", type=int, default=0, help="USB camera index")
    p.add_argument("--task", default="pick up the object and place it in the target")
    p.add_argument("--steps", type=int, default=5, help="Number of VLA inference steps")
    p.add_argument("--model", default="lerobot/smolvla_base")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--execute", action="store_true",
        help="Send actions to the robot. It WILL MOVE — ensure workspace is clear.",
    )
    p.add_argument(
        "--smooth-steps", type=int, default=20,
        help="Interpolation steps between current and target position (default 20).",
    )
    p.add_argument(
        "--step-delay", type=float, default=0.05,
        help="Seconds between interpolation steps (default 0.05 = 20 Hz).",
    )
    p.add_argument(
        "--calibrate", action="store_true",
        help="Run interactive motor calibration (needed once per robot setup).",
    )
    return p.parse_args()


def smooth_move(robot, current_deg: np.ndarray, target_deg: np.ndarray,
                n_steps: int, step_delay: float):
    """
    Linearly interpolate from current to target joint positions,
    sending each waypoint to the robot with a small delay.

    current_deg / target_deg: [6] float arrays in degrees.
    """
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        waypoint = current_deg + alpha * (target_deg - current_deg)
        action_dict = {f"{m}.pos": float(waypoint[j]) for j, m in enumerate(SO101_MOTORS)}
        robot._robot.send_action(action_dict)
        time.sleep(step_delay)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"[*] Device: {device}")

    print(f"[*] Loading SmolVLA ({args.model}) ...")
    vla = load_smolvla(model_id=args.model, device=device)
    print(f"[*] SmolVLA ready. hidden_dim={vla.get_hidden_dim()}, chunk={vla.get_chunk_size()}")

    env = SO101Env(
        task_description=args.task,
        port=args.port,
        camera_name="top",
        image_size=224,
        fps=30,
        calibrate=args.calibrate,
    )

    if args.execute:
        total_s = args.smooth_steps * args.step_delay
        print(f"\n  *** --execute ON — smooth motion: {args.smooth_steps} steps × "
              f"{args.step_delay}s = {total_s:.1f}s per action ***")
        print("  Ctrl+C to abort at any time. Pausing 3s...\n")
        time.sleep(3)

    try:
        print("[*] Resetting env (connecting hardware) ...")
        obs = env.reset()
        print("[*] Connected. Starting inference loop.\n")

        for step in range(1, args.steps + 1):
            t0 = time.perf_counter()
            out = vla(obs)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            action_chunk = out["action_chunk"]   # [1, chunk, 6]
            hidden = out["hidden_states"]         # [1, seq*layers, d_model]
            target = action_chunk[0, 0].cpu().numpy()   # first step of chunk [6]
            current = obs["observation.state"][0].cpu().numpy()  # current joints [6]

            print(f"[step {step}/{args.steps}]  inference: {elapsed_ms:.0f}ms")
            print(f"  current (deg) : {np.array2string(current, precision=1, suppress_small=True)}")
            print(f"  target  (deg) : {np.array2string(target,  precision=1, suppress_small=True)}")
            print(f"  delta   (deg) : {np.array2string(target - current, precision=1, suppress_small=True)}")
            print(f"  hidden_states : {tuple(hidden.shape)}")

            if args.execute:
                print(f"  [MOVING] interpolating over {args.smooth_steps} steps "
                      f"({args.smooth_steps * args.step_delay:.1f}s) ...")
                smooth_move(env, current, target, args.smooth_steps, args.step_delay)
                obs = env.reset()   # read fresh obs after motion completes
                print(f"  [DONE]")
            else:
                obs = env.reset()

            print()

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        env.close()
        print("[*] Disconnected cleanly.")


if __name__ == "__main__":
    main()
