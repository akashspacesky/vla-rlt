"""
Gym-compatible robot environment wrapper for RLT training.

Wraps a real robot interface or simulation (Isaac Lab) and provides:
- Reward functions for common precision manipulation tasks
- Human intervention detection (marks transitions as high-priority demos)
- Observation formatting compatible with GR00T N1 input spec

Task reward functions:
    - "sparse": +1.0 on success, 0 otherwise
    - "ethernet_insertion": shaped reward using end-effector force feedback
    - "screwdriver_alignment": shaped reward using visual alignment error

For a new task: subclass RobotEnv and override `_compute_reward`.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any


class RobotEnv(ABC):
    """
    Base class for robot environments compatible with groot-rlt training.

    Subclass this and implement:
        - reset() → obs dict
        - step(action_chunk) → (obs, reward, done, info)
        - _compute_reward(obs, action, next_obs, info) → float
    """

    def __init__(
        self,
        task_name: str,
        chunk_size: int = 16,
        action_dim: int = 7,
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
        """Reset environment. Returns GR00T-compatible obs dict."""
        ...

    @abstractmethod
    def _execute_action_chunk(self, action_chunk: np.ndarray) -> tuple[dict, dict]:
        """
        Execute action chunk on robot. Returns (next_obs, robot_info).
        robot_info may include: force_torque, success, human_override
        """
        ...

    def step(self, action_chunk: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Args:
            action_chunk: [chunk_size, action_dim] numpy array

        Returns:
            (next_obs, reward, done, info)
        """
        self._step_count += 1
        next_obs, robot_info = self._execute_action_chunk(action_chunk)

        reward = self._compute_reward(next_obs, action_chunk, robot_info)
        done = robot_info.get("success", False) or self._step_count >= self.max_episode_steps

        info = {
            "success": robot_info.get("success", False),
            "is_human_demo": robot_info.get("human_override", False),
            "step": self._step_count,
            **robot_info,
        }

        return next_obs, reward, done, info

    def _compute_reward(self, next_obs: dict, action_chunk: np.ndarray, info: dict) -> float:
        """Route to task-specific reward function."""
        if self.reward_mode == "sparse":
            return float(info.get("success", False))
        elif self.reward_mode == "ethernet_insertion":
            return self._reward_ethernet(next_obs, action_chunk, info)
        elif self.reward_mode == "screwdriver_alignment":
            return self._reward_screwdriver(next_obs, action_chunk, info)
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def _reward_ethernet(self, next_obs: dict, action_chunk: np.ndarray, info: dict) -> float:
        """
        Dense reward for ethernet insertion.
        Combines: sparse success + force-guided insertion progress + smoothness.

        Force feedback intuition: during successful insertion, Z-axis force
        increases then drops (click). Track this signature.
        """
        if info.get("success", False):
            return 1.0

        reward = 0.0

        # Insertion progress: positive Z-force indicates contact/insertion
        ft = info.get("force_torque", np.zeros(6))
        z_force = ft[2] if len(ft) > 2 else 0.0
        if 0.5 < z_force < 5.0:  # Reasonable insertion force range (N)
            reward += 0.1 * (z_force / 5.0)

        # Penalize large lateral forces (misalignment)
        lateral_force = np.sqrt(ft[0]**2 + ft[1]**2) if len(ft) > 1 else 0.0
        reward -= 0.05 * min(lateral_force / 2.0, 1.0)

        # Smoothness: penalize jerky actions
        action_jerk = float(np.diff(action_chunk, axis=0).max())
        reward -= 0.02 * min(action_jerk, 1.0)

        return float(np.clip(reward, -0.5, 0.5))

    def _reward_screwdriver(self, next_obs: dict, action_chunk: np.ndarray, info: dict) -> float:
        """
        Dense reward for screwdriver alignment.
        Uses visual alignment error from info dict (computed by vision pipeline).
        """
        if info.get("success", False):
            return 1.0

        reward = 0.0

        # Alignment error in pixels or mm from vision pipeline
        alignment_error = info.get("alignment_error_mm", None)
        if alignment_error is not None:
            # Shaped: higher reward as error decreases below threshold
            threshold_mm = 2.0
            if alignment_error < threshold_mm:
                reward += 0.3 * (1.0 - alignment_error / threshold_mm)

        # Contact force (positive = correct engagement)
        ft = info.get("force_torque", np.zeros(6))
        z_force = ft[2] if len(ft) > 2 else 0.0
        if 0.2 < z_force < 3.0:
            reward += 0.1

        return float(np.clip(reward, -0.5, 0.5))


class IsaacLabEnv(RobotEnv):
    """
    Wrapper for Isaac Lab simulation (for Sim2Real experiments).
    Requires Isaac Lab to be installed and configured.
    """

    def __init__(self, task_name: str, isaac_cfg: dict, **kwargs):
        super().__init__(task_name=task_name, **kwargs)
        self.isaac_cfg = isaac_cfg
        self._isaac_env = None  # Initialized lazily

    def _init_isaac(self):
        try:
            import isaaclab  # type: ignore
            # Initialize Isaac Lab environment
            # Specific setup depends on Isaac Lab version and task
            raise NotImplementedError(
                "Connect your Isaac Lab task here. See Isaac Lab docs for task registration."
            )
        except ImportError:
            raise ImportError("Isaac Lab not installed. See: https://isaac-sim.github.io/IsaacLab/")

    def reset(self) -> dict:
        if self._isaac_env is None:
            self._init_isaac()
        self._step_count = 0
        raw_obs = self._isaac_env.reset()
        return self._format_obs(raw_obs)

    def _execute_action_chunk(self, action_chunk: np.ndarray) -> tuple[dict, dict]:
        raw_obs, reward, done, info = self._isaac_env.step(action_chunk)
        return self._format_obs(raw_obs), info

    def _format_obs(self, raw_obs) -> dict:
        """Format Isaac Lab obs to GR00T input spec."""
        raise NotImplementedError("Implement obs formatting for your specific task.")


class MockRobotEnv(RobotEnv):
    """
    Mock environment for unit testing without a real robot or simulator.
    Produces random obs, rewards uniformly in [0, 1] with 10% success rate.
    """

    def __init__(self, **kwargs):
        super().__init__(task_name="mock", reward_mode="sparse", **kwargs)
        self._success_prob = 0.1

    def reset(self) -> dict:
        self._step_count = 0
        return self._random_obs()

    def _execute_action_chunk(self, action_chunk: np.ndarray) -> tuple[dict, dict]:
        next_obs = self._random_obs()
        success = np.random.random() < self._success_prob
        return next_obs, {"success": success, "force_torque": np.random.randn(6) * 0.5}

    def _random_obs(self) -> dict:
        return {
            "video": {"front_camera": np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8)},
            "state": np.random.randn(14).astype(np.float32),
            "annotation": {"human.validity": ["pick up the screwdriver and align it"]},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset for Phase 1 pre-training
# ─────────────────────────────────────────────────────────────────────────────

class DemoDataset(Dataset):
    """
    PyTorch dataset that loads GR00T-flavored LeRobot demos and pre-computes
    GR00T hidden states for RLTBottleneck pre-training.

    Caches hidden states to disk on first run to avoid repeated GR00T inference.
    Cache format: .pt files with tensors of shape [seq_len, d_model].

    Args:
        data_dir: Path to demo dataset (LeRobot v2 format)
        groot_wrapper: Initialized GR00TWrapperWithHooks
        cache_dir: Where to cache pre-extracted hidden states
        device: Torch device
    """

    def __init__(
        self,
        data_dir: str,
        groot_wrapper,
        cache_dir: str | None = None,
        device: str = "cpu",
    ):
        self.data_dir = Path(data_dir)
        self.groot_wrapper = groot_wrapper
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".rlt_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.device = device

        self.episode_paths = sorted(self.data_dir.glob("episode_*.pt"))
        if not self.episode_paths:
            raise FileNotFoundError(
                f"No episode files found in {data_dir}. "
                "Expected files named episode_*.pt in LeRobot v2 format."
            )

        print(f"[DemoDataset] Found {len(self.episode_paths)} episodes. Extracting hidden states...")
        self._extract_and_cache()

        self.hidden_state_files = sorted(self.cache_dir.glob("hs_*.pt"))
        print(f"[DemoDataset] Cached {len(self.hidden_state_files)} hidden state tensors.")

    def _extract_and_cache(self):
        """Extract hidden states from GR00T for each demo. Skip if already cached."""
        for ep_path in self.episode_paths:
            cache_path = self.cache_dir / f"hs_{ep_path.stem}.pt"
            if cache_path.exists():
                continue

            episode = torch.load(ep_path)
            # Episode is a list of obs dicts
            hs_list = []
            for obs in episode["observations"]:
                with torch.no_grad():
                    groot_out = self.groot_wrapper(obs)
                hs_list.append(groot_out["hidden_states"].cpu())

            torch.save(torch.stack(hs_list), cache_path)

    def __len__(self) -> int:
        return len(self.hidden_state_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load a random timestep from a random episode
        hs_ep = torch.load(self.hidden_state_files[idx])  # [T, seq*layers, d_model]
        t = np.random.randint(0, len(hs_ep))
        return {"hidden_states": hs_ep[t]}  # [seq*layers, d_model]
