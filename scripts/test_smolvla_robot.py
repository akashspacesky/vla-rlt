"""
SmolVLA inference smoke test on real SO-101.

Loads SmolVLA, connects SO-101 + USB cam, runs N inference steps.
Prints action chunks — does NOT move the robot unless --execute is passed.

Usage:
    # Observe only (safe):
    python3 scripts/test_smolvla_robot.py --port /dev/tty.usbmodem5A680100951

    # Actually move the robot (ensure workspace is clear):
    python3 scripts/test_smolvla_robot.py --port /dev/tty.usbmodem5A680100951 --execute

    # Custom task / more steps:
    python3 scripts/test_smolvla_robot.py --port /dev/tty.usbmodem5A680100951 \\
        --task "pick up the red block" --steps 10
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
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--model", default="lerobot/smolvla_base")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--execute", action="store_true",
        help="Send actions to the robot. It WILL MOVE — ensure workspace is clear.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"[*] Device: {device}")

    # Load SmolVLA
    print(f"[*] Loading SmolVLA ({args.model}) ...")
    vla = load_smolvla(model_id=args.model, device=device)
    print(f"[*] SmolVLA ready. hidden_dim={vla.get_hidden_dim()}, chunk={vla.get_chunk_size()}")

    # Connect SO-101 + camera
    env = SO101Env(
        task_description=args.task,
        port=args.port,
        camera_name="top",
        image_size=224,
        fps=30,
    )

    if args.execute:
        print("\n  *** --execute ON — robot WILL MOVE. Ctrl+C to abort. Pausing 3s... ***\n")
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
            first = action_chunk[0, 0].cpu().numpy()

            state = obs["observation.state"][0].cpu().numpy()
            print(f"[step {step}/{args.steps}]  inference: {elapsed_ms:.0f}ms")
            print(f"  state (deg)        : {np.array2string(state, precision=1, suppress_small=True)}")
            print(f"  action_chunk shape : {tuple(action_chunk.shape)}")
            print(f"  hidden_states shape: {tuple(hidden.shape)}")
            print(f"  first action (deg) : {np.array2string(first, precision=2, suppress_small=True)}")

            if args.execute:
                obs, reward, done, info = env.step(action_chunk[0].cpu().numpy())
                print(f"  [SENT] reward={reward:.3f}  done={done}")
                if done:
                    print("[*] Episode done — resetting.")
                    obs = env.reset()
            else:
                # Refresh obs without executing
                obs = env.reset()

            print()

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        env.close()
        print("[*] Disconnected cleanly.")


if __name__ == "__main__":
    main()
