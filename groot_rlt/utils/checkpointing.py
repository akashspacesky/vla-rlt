"""Checkpointing: save/load model states with rotation."""

import torch
from pathlib import Path
from typing import Any


class Checkpointer:
    def __init__(self, save_dir: str | Path, keep_last_n: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._saved: list[Path] = []

    def save(self, state: dict[str, Any], name: str, step: int):
        path = self.save_dir / f"{name}_step{step:07d}.pt"
        torch.save(state, path)
        self._saved.append(path)

        # Also save a "latest" symlink for convenience
        latest = self.save_dir / f"{name}_latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)

        # Rotate old checkpoints
        named = [p for p in self._saved if p.stem.startswith(name) and "step" in p.stem]
        named.sort(key=lambda p: int(p.stem.split("step")[-1]))
        while len(named) > self.keep_last_n:
            old = named.pop(0)
            if old.exists():
                old.unlink()

    def load_latest(self, name: str) -> dict[str, Any] | None:
        latest = self.save_dir / f"{name}_latest.pt"
        if not latest.exists():
            return None
        return torch.load(latest, map_location="cpu")
