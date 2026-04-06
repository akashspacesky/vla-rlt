"""
RLTTrainer: Orchestrates both training phases.

Phase 1 — Offline Bottleneck Pre-training:
    Load GR00T, collect/load demonstration data, extract hidden states,
    train RLTBottleneck encoder-decoder to reconstruct them. Saves encoder checkpoint.

Phase 2 — Online SAC:
    Load pre-trained encoder (frozen). Run SAC on robot. Actor + critic update
    every step for `updates_per_step` gradient steps (enables fast on-robot learning).
    Anneals reference-action dropout: high → 0 → plateau at ref_dropout_prob.
"""

import time
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig

from groot_rlt.models import RLTBottleneck, RLTActor, RLTCritic
from groot_rlt.models.groot_wrapper import GR00TWrapperWithHooks
from groot_rlt.training.sac import SAC
from groot_rlt.training.replay_buffer import ReplayBuffer
from groot_rlt.utils.logging import RLTLogger
from groot_rlt.utils.checkpointing import Checkpointer


class RLTTrainer:
    """
    Main training orchestrator for groot-rlt.

    Args:
        cfg: Hydra/OmegaConf config (see configs/default.yaml)
        groot_wrapper: GR00T model with hidden-state hooks
        env: Gym-compatible robot environment
    """

    def __init__(
        self,
        cfg: DictConfig,
        groot_wrapper: GR00TWrapperWithHooks,
        env,
    ):
        self.cfg = cfg
        self.groot_wrapper = groot_wrapper
        self.env = env
        self.device = cfg.device

        d_model = groot_wrapper.get_hidden_dim()

        # Build models
        self.bottleneck = RLTBottleneck(
            d_model=d_model,
            d_rlt=cfg.rlt_dim,
            seq_len=cfg.seq_len,
            num_heads=cfg.encoder.num_heads,
            encoder_layers=cfg.encoder.num_layers,
            decoder_layers=cfg.decoder.num_layers,
            dropout=cfg.encoder.dropout,
            num_vlm_layers_to_use=cfg.num_hook_layers,
        ).to(self.device)

        self.actor = RLTActor(
            d_rlt=cfg.rlt_dim,
            action_dim=cfg.action_dim,
            chunk_size=cfg.chunk_size,
            hidden_dim=cfg.actor.hidden_dim,
            num_layers=cfg.actor.num_layers,
            ref_dropout_prob=cfg.ref_dropout_prob,
            action_scale=cfg.action_scale,
        ).to(self.device)

        self.critic = RLTCritic(
            d_rlt=cfg.rlt_dim,
            action_dim=cfg.action_dim,
            chunk_size=cfg.chunk_size,
            hidden_dim=cfg.critic.hidden_dim,
            num_layers=cfg.critic.num_layers,
        ).to(self.device)

        self.sac = SAC(
            actor=self.actor,
            critic=self.critic,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            alpha_lr=cfg.alpha_lr,
            gamma=cfg.gamma,
            tau=cfg.tau,
            init_alpha=cfg.init_alpha,
            ref_reg_weight=cfg.ref_reg_weight,
            gradient_clip=cfg.gradient_clip,
            device=self.device,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=cfg.buffer_capacity,
            d_rlt=cfg.rlt_dim,
            action_dim=cfg.action_dim,
            chunk_size=cfg.chunk_size,
            human_demo_ratio=cfg.human_demo_ratio,
            device=self.device,
        )

        self.logger = RLTLogger(
            project=cfg.wandb.project,
            name=cfg.run_name,
            cfg=cfg,
            use_wandb=cfg.wandb.enabled,
        )

        self.checkpointer = Checkpointer(
            save_dir=Path(cfg.checkpoint_dir) / cfg.run_name,
            keep_last_n=cfg.keep_checkpoints,
        )

        self._step = 0
        self._episode = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Offline Bottleneck Pre-training
    # ─────────────────────────────────────────────────────────────────────────

    def pretrain(self, demo_data_dir: str, max_epochs: int = 50):
        """
        Phase 1: Train the RLTBottleneck on offline demonstration data.

        Loads demos, extracts GR00T hidden states for each, trains encoder-decoder
        with MSE reconstruction loss.

        Args:
            demo_data_dir: Path to GR00T-flavored LeRobot v2 dataset directory
            max_epochs: Number of training epochs
        """
        from torch.utils.data import DataLoader
        from groot_rlt.envs.robot_env import DemoDataset

        print(f"[Phase 1] Pre-training RLT bottleneck on demos from {demo_data_dir}")

        dataset = DemoDataset(demo_data_dir, self.groot_wrapper, device=self.device)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.pretrain_batch_size,
            shuffle=True,
            num_workers=0,  # Keep in-process for robot hardware
        )

        optimizer = torch.optim.AdamW(
            self.bottleneck.parameters(),
            lr=self.cfg.pretrain_lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_loss = float("inf")
        for epoch in range(max_epochs):
            total_loss = 0.0
            for batch in dataloader:
                hidden_states = batch["hidden_states"].to(self.device)
                result = self.bottleneck(hidden_states)
                loss = result["loss"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.bottleneck.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            scheduler.step()

            self.logger.log({"pretrain/recon_loss": avg_loss, "pretrain/epoch": epoch})
            print(f"  Epoch {epoch+1}/{max_epochs}  recon_loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.checkpointer.save(
                    {"bottleneck": self.bottleneck.state_dict()},
                    name="bottleneck_best",
                    step=epoch,
                )

        print(f"[Phase 1] Done. Best recon loss: {best_loss:.4f}")
        return best_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Online SAC
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, total_steps: int, warm_up_steps: int = 1000):
        """
        Phase 2: Online SAC training on the robot.

        Runs `updates_per_step` SAC gradient steps after every environment step.
        Encoder stays frozen. Only actor and critic are updated.

        Args:
            total_steps: Total environment steps to run
            warm_up_steps: Steps before SAC updates begin (random exploration)
        """
        # Freeze encoder — only actor/critic update during online RL
        for p in self.bottleneck.encoder.parameters():
            p.requires_grad_(False)

        print(f"[Phase 2] Online SAC: {total_steps} steps, warm-up={warm_up_steps}")
        obs = self.env.reset()
        episode_reward = 0.0
        episode_start = time.time()

        for step in range(total_steps):
            self._step = step

            # Anneal reference-action dropout
            dropout_prob = self._anneal_ref_dropout(step)
            self.actor.set_ref_dropout_prob(dropout_prob)

            # Get GR00T hidden states and VLA action chunk
            groot_out = self.groot_wrapper(obs)
            rl_token = self.bottleneck.encode(
                groot_out["hidden_states"].unsqueeze(0).to(self.device)
            ).squeeze(0)
            vla_action = groot_out["action_chunk"].squeeze(0).to(self.device)

            # Select action
            if step < warm_up_steps:
                # Random exploration: small perturbation around VLA action
                delta = torch.randn_like(vla_action) * self.cfg.action_scale * 0.5
                action_chunk = vla_action + delta
            else:
                action_chunk = self.actor.get_action(
                    rl_token.unsqueeze(0),
                    vla_action.unsqueeze(0),
                    deterministic=False,
                ).squeeze(0)

            delta = action_chunk - vla_action

            # Step environment
            next_obs, reward, done, info = self.env.step(action_chunk.cpu().numpy())
            episode_reward += reward

            # Get next RL token
            next_groot_out = self.groot_wrapper(next_obs)
            next_rl_token = self.bottleneck.encode(
                next_groot_out["hidden_states"].unsqueeze(0).to(self.device)
            ).squeeze(0)

            # Store transition
            self.replay_buffer.add(
                rl_token=rl_token.cpu(),
                vla_action=vla_action.cpu(),
                delta_action=delta.cpu(),
                reward=reward,
                next_rl_token=next_rl_token.cpu(),
                done=done,
                is_human_demo=info.get("is_human_demo", False),
            )

            # SAC updates
            if step >= warm_up_steps and self.replay_buffer.is_ready(self.cfg.min_buffer_size):
                update_logs = {}
                for _ in range(self.cfg.updates_per_step):
                    batch = self.replay_buffer.sample(self.cfg.batch_size)
                    update_logs = self.sac.update(batch)
                self.logger.log({f"train/{k}": v for k, v in update_logs.items()}, step=step)

            # Episode end
            if done:
                self._episode += 1
                episode_time = time.time() - episode_start
                self.logger.log({
                    "train/episode_reward": episode_reward,
                    "train/episode_length": step - (self._step - episode_reward),  # approx
                    "train/episode_time_s": episode_time,
                    "train/episodes": self._episode,
                    "train/ref_dropout_prob": dropout_prob,
                    "train/buffer_size": len(self.replay_buffer),
                }, step=step)
                print(
                    f"  Step {step} | Episode {self._episode} | "
                    f"Reward {episode_reward:.2f} | α={self.sac.alpha.item():.3f}"
                )
                obs = self.env.reset()
                episode_reward = 0.0
                episode_start = time.time()
            else:
                obs = next_obs

            # Checkpoint
            if step % self.cfg.checkpoint_every == 0:
                self.checkpointer.save(
                    {
                        "sac": self.sac.state_dict(),
                        "bottleneck_encoder": self.bottleneck.encoder.state_dict(),
                        "step": step,
                    },
                    name="latest",
                    step=step,
                )

        print(f"[Phase 2] Training complete. Total SAC updates: {self.sac.total_updates}")

    def _anneal_ref_dropout(self, step: int) -> float:
        """
        Anneal reference-action dropout probability.
        Schedule: 0.5 → 0.0 over first `dropout_warmup` steps,
                  then back up to cfg.ref_dropout_prob over next `dropout_warmup` steps,
                  then constant.
        This ensures actor learns both with and without VLA guidance.
        """
        warmup = self.cfg.get("dropout_warmup_steps", 1000)
        target = self.cfg.ref_dropout_prob
        if step < warmup:
            # Linear decay from 0.5 to 0.0
            return 0.5 * (1.0 - step / warmup)
        elif step < 2 * warmup:
            # Linear rise from 0.0 to target
            return target * (step - warmup) / warmup
        else:
            return target

    def load_pretrained_encoder(self, checkpoint_path: str):
        """Load encoder weights from Phase 1 checkpoint."""
        state = torch.load(checkpoint_path, map_location=self.device)
        if "bottleneck" in state:
            self.bottleneck.load_state_dict(state["bottleneck"])
        elif "bottleneck_encoder" in state:
            self.bottleneck.encoder.load_state_dict(state["bottleneck_encoder"])
        print(f"Loaded encoder from {checkpoint_path}")

    def load_sac(self, checkpoint_path: str):
        """Load SAC (actor+critic) from checkpoint."""
        state = torch.load(checkpoint_path, map_location=self.device)
        self.sac.load_state_dict(state["sac"])
        print(f"Loaded SAC from {checkpoint_path}")
