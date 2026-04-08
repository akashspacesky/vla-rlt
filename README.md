# vla-rlt

**RL Tokens for open-source VLAs** — Online reinforcement learning adaptation of [SmolVLA](https://huggingface.co/lerobot/smolvla_base) for precise robot manipulation on the SO-101, inspired by Physical Intelligence's [RLT paper](https://www.pi.website/research/rlt).

Runs on **Apple Silicon (MPS)** — no GPU server needed.

## Core Idea

Pre-trained VLAs are good at general manipulation but lack precision for tight-tolerance tasks. RLT adds a compact **RL token** — a bottleneck of the VLA's internal state — that a small actor-critic learns over online with ~15 minutes of real robot data.

```
Images + Language
      │
 [SmolVLA VLM]  ← frozen
      │
 hidden states
      │
 [RLT Encoder] → rl_token (d=128)
                        │
               [Actor MLP] + vla_action → delta_action
                        │
           final_action = vla_action + delta
```

The actor **edits** VLA actions (additive delta), never replacing them.

## Results (Pi paper on similar approach)

| Task | Baseline | RLT | Speedup |
|------|----------|-----|---------|
| Ethernet insertion | 147/10min | 400/10min | **2.7x** |
| Screwdriver alignment | — | — | **3x** |

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| `VLABackend` | `models/vla_backend.py` | Abstract interface — plug in any VLA |
| `SmolVLAWrapper` | `models/smolvla_wrapper.py` | SmolVLA with hidden-state hooks |
| `RLTEncoder` | `models/rlt_encoder.py` | Bottleneck transformer → rl_token |
| `RLTActor` | `models/actor.py` | SAC actor: tanh-squashed action delta |
| `RLTCritic` | `models/critic.py` | Twin-Q SAC critic |
| `SAC` | `training/sac.py` | SAC + reference regularization, MPS-aware |
| `RLTTrainer` | `training/trainer.py` | Two-phase: offline pretrain → online SAC |
| `SO101Env` | `envs/robot_env.py` | Real SO-101 via LeRobot |

## Quickstart

```bash
# 0. Install uv
curl -Ls https://astral.sh/uv/install.sh | sh

# 1. Clone and install
git clone https://github.com/akashspacesky/vla-rlt
cd vla-rlt
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Validate (no hardware needed)
python scripts/pretrain_rlt.py --dry_run
pytest tests/ -v

# 3. Phase 1: pre-train on demos
python scripts/pretrain_rlt.py \
    --config configs/default.yaml \
    --data_dir /path/to/lerobot_demos

# 4. Phase 2: online SAC on SO-101
uv pip install lerobot  # adds robot drivers
python scripts/train.py \
    --config configs/tasks/so101_pick_place.yaml \
    --encoder_ckpt checkpoints/bottleneck_pretrained.pt

# 5. Evaluate
python scripts/evaluate.py \
    --checkpoint runs/rlt_so101_pick_place/sac_latest.pt \
    --num_episodes 20 --compare_baseline
```

## Key Hyperparameters

| Param | Default | Effect |
|-------|---------|--------|
| `rlt_dim` | 128 | RL token size |
| `ref_reg_weight` | 0.1 | L2 penalty toward VLA action |
| `updates_per_step` | 50 | SAC steps per env step (tuned for MPS) |
| `action_scale` | 0.1 | Max delta magnitude |
| `device` | auto | cuda > mps > cpu |

## Adding a New VLA

Subclass `VLABackend` in `rlt/models/` and implement 4 methods. See `smolvla_wrapper.py` as reference.

## Roadmap

- [ ] Dense reward from force-torque
- [ ] Multi-task: shared encoder, task-conditioned actor
- [ ] Sim2Real via Isaac Lab / MuJoCo
- [ ] Hook SmolVLA's action expert layers (not just VLM)
- [ ] OpenVLA backend

## Citation

```bibtex
@misc{pi2024rlt,
  title={RLT: Reinforcement Learning Tokens for Precise Manipulation},
  author={Physical Intelligence},
  year={2024},
  url={https://www.pi.website/research/rlt}
}
```

## License

MIT
