"""Logging: WandB + console for RLT training."""

import time
from typing import Any
from omegaconf import OmegaConf, DictConfig


class RLTLogger:
    def __init__(
        self,
        project: str,
        name: str,
        cfg: DictConfig | None = None,
        use_wandb: bool = True,
    ):
        self.use_wandb = use_wandb
        self._wandb = None
        self._last_print = time.time()

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                cfg_dict = OmegaConf.to_container(cfg, resolve=True) if cfg else {}
                wandb.init(project=project, name=name, config=cfg_dict)
            except ImportError:
                print("[RLTLogger] wandb not installed, falling back to console only.")
                self.use_wandb = False

    def log(self, data: dict[str, Any], step: int | None = None):
        if self._wandb and self.use_wandb:
            self._wandb.log(data, step=step)

        # Throttle console output to every 5 seconds
        now = time.time()
        if now - self._last_print > 5.0:
            scalars = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in data.items()}
            step_str = f"[step {step}] " if step is not None else ""
            print(f"  {step_str}" + "  ".join(f"{k}={v}" for k, v in scalars.items()))
            self._last_print = now

    def finish(self):
        if self._wandb and self.use_wandb:
            self._wandb.finish()
