"""
Robot environments for RLT training.

SO101Env: Concrete env for the SO-101 arm via LeRobot's SOFollower interface.

Observation format (SmolVLA-compatible):
    {
        "observation.images.<cam>": [1, C, H, W]  float32 in [0, 1]
        "observation.state":        [1, state_dim] float32
        "task":                     ["pick up the red block"]
    }

SO-101 motor order (matches action_dim=6):
    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

Action sent to hardware:
    {"shoulder_pan.pos": v0, "shoulder_lift.pos": v1, ..., "gripper.pos": v5}
    (SOFollower.send_action strips the ".pos" suffix internally)
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from torch.utils.data import Dataset


# SO-101 motor names in canonical order (matches 6-DoF action_dim)
SO101_MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class RobotEnv(ABC):
    def __init__(
        self,
        task_name: str,
        chunk_size: int = 16,
        action_dim: int = 6,
        max_episode_steps: int = 200,
        reward_mode: str = "sparse",
    ):
        self.task_name = task_name
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.reward_mode = reward_mode
        self._step_count = 0

    @abstractmethod
    def reset(self) -> dict:
        ...

    @abstractmethod
    def _execute_action_chunk(self, action_chunk: np.ndarray) -> tuple[dict, dict]:
        ...

    def step(self, action_chunk: np.ndarray) -> tuple[dict, float, bool, dict]:
        self._step_count += 1
        next_obs, info = self._execute_action_chunk(action_chunk)
        reward = self._compute_reward(next_obs, action_chunk, info)
        done = info.get("success", False) or self._step_count >= self.max_episode_steps
        info.update({"step": self._step_count, "success": info.get("success", False)})
        return next_obs, reward, done, info

    def _compute_reward(self, next_obs, action_chunk, info) -> float:
        if self.reward_mode == "sparse":
            return float(info.get("success", False))
        elif self.reward_mode == "so101_pick_place":
            return self._reward_pick_place(next_obs, action_chunk, info)
        raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def _reward_pick_place(self, next_obs, action_chunk, info) -> float:
        if info.get("success", False):
            return 1.0
        reward = 0.0
        state = info.get("robot_state", np.zeros(6))
        gripper_norm = state[5] / 100.0 if len(state) > 5 else 0.0
        if 0.3 < gripper_norm < 0.9:
            reward += 0.1
        joints = state[:5] / 100.0 if len(state) >= 5 else np.zeros(5)
        joint_margin = np.minimum(joints + 1.0, 1.0 - joints).min()
        if joint_margin < 0.1:
            reward -= 0.05
        return float(np.clip(reward, -0.2, 0.5))


class SO101Env(RobotEnv):
    """
    Real SO-101 robot environment.

    Motor control via LeRobot SOFollower; camera via cv2 directly
    (avoids av/cv2 libavdevice dylib conflict on macOS).

    Args:
        task_description: Natural language task instruction for SmolVLA.
        port: USB serial port, e.g. "/dev/tty.usbmodem..."
        camera_index: cv2 camera index (default 0).
        camera_name: Key name used in observation dict (default "top").
        image_size: Resize frames to this square size (default 224).
        fps: Target control frequency (default 30).
        use_degrees: Degree normalization for motors (default True).
        calibrate: Run interactive calibration on connect (default False).
        robot: Pre-instantiated SOFollower (skips internal setup).
    """

    def __init__(
        self,
        task_description: str = "pick up the object and place it in the target",
        port: str = "/dev/ttyUSB0",
        camera_index: int = 0,
        camera_name: str = "top",
        image_size: int = 224,
        fps: int = 30,
        use_degrees: bool = True,
        calibrate: bool = False,
        robot=None,
        **kwargs,
    ):
        super().__init__(task_name="so101", action_dim=6, **kwargs)
        self.task_description = task_description
        self.port = port
        self.camera_index = camera_index
        self.camera_name = camera_name
        self.image_size = image_size
        self.fps = fps
        self.use_degrees = use_degrees
        self.calibrate = calibrate
        self._robot = robot
        self._cap = None

    def _init_hardware(self):
        import cv2
        try:
            from lerobot.robots.so_follower.so_follower import SOFollower
            from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        except ImportError:
            raise ImportError(
                "LeRobot not installed. Run:\n"
                "  uv pip install lerobot feetech-servo-sdk"
            )

        # Motor-only config — camera handled by cv2 to avoid av/cv2 dylib conflict
        config = SOFollowerRobotConfig(
            port=self.port,
            cameras={},
            use_degrees=self.use_degrees,
        )
        print(f"[SO101Env] Connecting motors on {self.port} ...")
        self._robot = SOFollower(config)
        self._robot.connect(calibrate=self.calibrate)

        print(f"[SO101Env] Opening camera {self.camera_index} ...")
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        print("[SO101Env] Connected.")

    def reset(self) -> dict:
        if self._robot is None:
            self._init_hardware()
        self._step_count = 0
        return self._get_obs()

    def _execute_action_chunk(self, action_chunk: np.ndarray) -> tuple[dict, dict]:
        action_vec = action_chunk[0]
        action_dict = {
            f"{motor}.pos": float(action_vec[i])
            for i, motor in enumerate(SO101_MOTORS)
        }
        self._robot.send_action(action_dict)
        next_obs = self._get_obs()
        robot_state = self._obs_to_state_array(self._robot.get_observation())
        success = self._check_success(robot_state)
        return next_obs, {"success": success, "robot_state": robot_state}

    def _get_obs(self) -> dict:
        import cv2
        import torch.nn.functional as F

        raw = self._robot.get_observation()
        state = torch.tensor(
            self._obs_to_state_array(raw), dtype=torch.float32
        ).unsqueeze(0)  # [1, 6]

        # Camera via cv2 (avoids av/cv2 dylib conflict in lerobot camera thread)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("cv2 camera read failed")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [C, H, W]
        if img.shape[1] != self.image_size or img.shape[2] != self.image_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return {
            f"observation.images.{self.camera_name}": img.unsqueeze(0),  # [1,C,H,W]
            "observation.state": state,                                    # [1, 6]
            "task": [self.task_description],
        }

    def _obs_to_state_array(self, raw_obs: dict) -> np.ndarray:
        return np.array(
            [raw_obs.get(f"{m}.pos", 0.0) for m in SO101_MOTORS],
            dtype=np.float32,
        )

    def _check_success(self, state: np.ndarray) -> bool:
        return False

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._robot is not None and self._robot.is_connected:
            self._robot.disconnect()
            self._robot = None


class MockRobotEnv(RobotEnv):
    """Mock env for testing. No hardware needed. Matches SO-101 / SmolVLA format."""

    def __init__(
        self,
        success_prob: float = 0.1,
        image_size: int = 224,
        camera_name: str = "top",
        **kwargs,
    ):
        super().__init__(task_name="mock", reward_mode="sparse", action_dim=6, **kwargs)
        self.success_prob = success_prob
        self.image_size = image_size
        self.camera_name = camera_name

    def reset(self) -> dict:
        self._step_count = 0
        return self._random_obs()

    def _execute_action_chunk(self, action_chunk: np.ndarray) -> tuple[dict, dict]:
        success = np.random.random() < self.success_prob
        return self._random_obs(), {"success": success, "robot_state": np.random.randn(6)}

    def _random_obs(self) -> dict:
        return {
            f"observation.images.{self.camera_name}": torch.rand(
                1, 3, self.image_size, self.image_size
            ),
            "observation.state": torch.randn(1, 6),
            "task": ["pick up the block"],
        }


class DemoDataset(Dataset):
    """
    Dataset for Phase 1 pre-training.
    Each episode file (episode_*.pt) should be a dict:
        {"observations": [obs_dict, ...], "actions": [...]}
    """

    def __init__(self, data_dir: str, vla, cache_dir: str | None = None, device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.vla = vla
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".rlt_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.device = device

        self.episode_paths = sorted(self.data_dir.glob("episode_*.pt"))
        if not self.episode_paths:
            raise FileNotFoundError(f"No episode_*.pt files in {data_dir}")

        print(f"[DemoDataset] Extracting hidden states from {len(self.episode_paths)} episodes...")
        self._extract_and_cache()
        self.hs_files = sorted(self.cache_dir.glob("hs_*.pt"))
        print(f"[DemoDataset] Ready. {len(self.hs_files)} cached tensors.")

    def _extract_and_cache(self):
        for ep_path in self.episode_paths:
            cache_path = self.cache_dir / f"hs_{ep_path.stem}.pt"
            if cache_path.exists():
                continue
            episode = torch.load(ep_path)
            hs_list = []
            for obs in episode["observations"]:
                with torch.no_grad():
                    out = self.vla(obs)
                hs_list.append(out["hidden_states"].cpu())
            torch.save(torch.stack(hs_list), cache_path)

    def __len__(self) -> int:
        return len(self.hs_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hs_ep = torch.load(self.hs_files[idx])
        t = np.random.randint(0, len(hs_ep))
        return {"hidden_states": hs_ep[t]}
