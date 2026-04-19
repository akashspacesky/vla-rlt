#!/usr/bin/env python
"""
Demo collection for SO-101 via passive teleoperation (gravity-compensation mode).

Strategy:
  1. Connect to the SO-101 follower robot.
  2. Disable torque -- user physically moves the arm.
  3. Record joint positions + camera frames at control_hz.
  4. Press ENTER to end the episode; press Ctrl-C to stop collection.
  5. Each episode saved as episode_NNNN.pt in --out_dir.

Episode format (for Phase 1 pretraining):
    {
      "observations": [obs_dict, ...],   # one per timestep
      "actions":      [action_vec, ...], # np.ndarray [action_dim]
      "task":         "task description string",
      "fps":          int,
    }
Each obs_dict:
    {
      "observation.images.top": [1, C, H, W] float32 tensor in [0,1],
      "observation.state":      [1, 6] float32 tensor,
      "task":                   [str],
    }

Usage:
    python scripts/collect_demos.py \\
        --port /dev/tty.usbserial-FT1234 \\
        --task "pick up the lego brick and place it in the box" \\
        --out_dir data/demos/pick_place \\
        --num_episodes 20 \\
        --fps 10
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import numpy as np
import torch


SO101_MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args():
    p = argparse.ArgumentParser(description="Collect SO-101 demos via passive teleoperation.")
    p.add_argument("--port", default="/dev/ttyUSB0",
                   help="Serial port for SO-101 (e.g. /dev/tty.usbserial-FT1234)")
    p.add_argument("--task", required=True,
                   help="Natural language task description (used as VLA prompt)")
    p.add_argument("--out_dir", required=True,
                   help="Directory to save episode files (created if needed)")
    p.add_argument("--num_episodes", type=int, default=20,
                   help="Number of episodes to collect")
    p.add_argument("--fps", type=int, default=10,
                   help="Recording frequency in Hz")
    p.add_argument("--camera_name", default="top",
                   help="Camera key in SOFollower config")
    p.add_argument("--image_size", type=int, default=224,
                   help="Resize camera output to NxN")
    p.add_argument("--max_steps", type=int, default=200,
                   help="Max steps per episode (auto-stops at limit)")
    p.add_argument("--dry_run", action="store_true",
                   help="Run with MockRobotEnv instead of real hardware")
    return p.parse_args()


def connect_robot(args):
    try:
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    except ImportError:
        print("ERROR: LeRobot not installed.")
        print("  git clone https://github.com/huggingface/lerobot")
        print("  uv pip install -e 'lerobot/[smolvla]'")
        sys.exit(1)

    cameras = {
        args.camera_name: OpenCVCameraConfig(
            index=0,
            fps=args.fps,
            width=args.image_size,
            height=args.image_size,
        )
    }
    config = SOFollowerRobotConfig(
        port=args.port,
        cameras=cameras,
        use_degrees=True,
        disable_torque_on_disconnect=True,
    )
    robot = SOFollower(config)
    robot.connect(calibrate=False)
    return robot


def read_state(raw_obs: dict) -> np.ndarray:
    """Extract motor positions in canonical order -> [6] float32."""
    return np.array(
        [raw_obs.get(f"{m}.pos", 0.0) for m in SO101_MOTORS],
        dtype=np.float32,
    )


def raw_obs_to_rlt(raw_obs: dict, camera_name: str, image_size: int, task: str) -> dict:
    """Convert SOFollower.get_observation() dict -> SmolVLA-compatible obs dict."""
    state = torch.tensor(read_state(raw_obs), dtype=torch.float32).unsqueeze(0)  # [1, 6]

    frame = raw_obs.get(camera_name)
    if frame is None:
        raise KeyError(f"Camera '{camera_name}' not in observation. Keys: {list(raw_obs.keys())}")
    img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [C, H, W]
    if img.shape[1] != image_size or img.shape[2] != image_size:
        import torch.nn.functional as F
        img = F.interpolate(
            img.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return {
        f"observation.images.{camera_name}": img.unsqueeze(0),  # [1,C,H,W]
        "observation.state": state,                               # [1, 6]
        "task": [task],
    }


def collect_episode_real(robot, args) -> dict | None:
    """
    Record one episode in passive mode (torque off = user moves arm freely).
    Returns episode dict or None if episode was discarded.
    """
    print("\n--- New Episode ---")
    print("Torque disabled. Move the arm to your starting position.")
    print("Press ENTER to start recording, or 'q'+ENTER to quit.")

    user_in = input("> ").strip().lower()
    if user_in == "q":
        return None

    # Disable torque for passive recording
    robot.bus.disable_torque()
    print(f"Recording at {args.fps} Hz. Press ENTER to stop (max {args.max_steps} steps).")

    # Non-blocking ENTER detection
    stop_flag = threading.Event()

    def wait_for_enter():
        input()
        stop_flag.set()

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    observations = []
    actions = []
    dt = 1.0 / args.fps

    step = 0
    while not stop_flag.is_set() and step < args.max_steps:
        t0 = time.perf_counter()

        raw = robot.get_observation()
        state = read_state(raw)
        obs = raw_obs_to_rlt(raw, args.camera_name, args.image_size, args.task)

        observations.append(obs)
        actions.append(state.copy())  # in passive mode, current state = intended action

        step += 1
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, dt - elapsed))

    # Re-enable torque after recording
    robot.bus.enable_torque()
    stop_flag.set()

    print(f"Recorded {step} steps.")
    if step < 5:
        print("Too short -- discarding.")
        return None

    print("Keep this episode? [y/n]")
    if input("> ").strip().lower() != "y":
        print("Discarded.")
        return None

    return {
        "observations": observations,
        "actions": [a for a in actions],
        "task": args.task,
        "fps": args.fps,
    }


def collect_episode_mock(args) -> dict:
    """Generate a fake episode for dry-run testing."""
    steps = min(30, args.max_steps)
    observations = []
    actions = []
    for _ in range(steps):
        obs = {
            f"observation.images.{args.camera_name}": torch.rand(1, 3, args.image_size, args.image_size),
            "observation.state": torch.randn(1, 6),
            "task": [args.task],
        }
        observations.append(obs)
        actions.append(np.random.randn(6).astype(np.float32))
        time.sleep(1.0 / args.fps)
    return {
        "observations": observations,
        "actions": actions,
        "task": args.task,
        "fps": args.fps,
    }


def save_episode(episode: dict, out_dir: Path, episode_idx: int):
    path = out_dir / f"episode_{episode_idx:04d}.pt"
    torch.save(episode, path)
    print(f"Saved: {path}  ({len(episode['observations'])} steps)")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find next episode index
    existing = sorted(out_dir.glob("episode_*.pt"))
    next_idx = len(existing)

    print(f"Task: {args.task}")
    print(f"Output: {out_dir}")
    print(f"Episodes to collect: {args.num_episodes} (starting at #{next_idx})")

    if args.dry_run:
        print("\n[DRY RUN] Using mock robot -- no hardware needed.")
        for i in range(args.num_episodes):
            print(f"\nCollecting mock episode {next_idx + i + 1}/{next_idx + args.num_episodes} ...")
            ep = collect_episode_mock(args)
            save_episode(ep, out_dir, next_idx + i)
        print(f"\nDone. Collected {args.num_episodes} mock episodes.")
        return

    robot = connect_robot(args)
    print(f"Connected to SO-101 on {args.port}.")

    collected = 0
    try:
        while collected < args.num_episodes:
            print(f"\nEpisode {next_idx + collected + 1} / {next_idx + args.num_episodes}")
            ep = collect_episode_real(robot, args)
            if ep is None:
                break
            save_episode(ep, out_dir, next_idx + collected)
            collected += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if robot.is_connected:
            robot.bus.enable_torque()
            robot.disconnect()
            print("Robot disconnected.")

    print(f"\nDone. Collected {collected} episodes in {out_dir}")


if __name__ == "__main__":
    main()
