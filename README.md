# groot-rlt

**RL Tokens for GR00T** — Online reinforcement learning adaptation of NVIDIA's [GR00T N1](https://github.com/NVIDIA/Isaac-GR00T) VLA for precise robot manipulation, inspired by Physical Intelligence's [RLT paper](https://www.pi.website/research/rlt).

## Core Idea

Pre-trained VLAs excel at general manipulation but struggle with tight-tolerance tasks (ethernet insertion, screwdriver alignment). RLT adds a compact **RL token** — a bottleneck representation of the VLA's internal state — that a small actor-critic can learn over online with ~15 minutes of real-world robot data.

```
Images + Language
      │
 [GR00T N1 VLM]  ← frozen
      │
 hidden states
      │
 [RLT Encoder] → rl_token (d=256)
                        │
               [Actor MLP] + vla_action → delta_action
                        │
           final_action = vla_action + delta
```

The actor **edits** the VLA's predicted action (additive delta), never replacing it. This preserves learned manipulation priors while RL refines precision.

## Results (Pi paper on similar approach)

| Task | Baseline | RLT | Speedup |
|------|----------|-----|---------|
| Ethernet insertion | 147/10min | 400/10min | **2.7x** |
| Screwdriver alignment | — | — | **3x** |

Achieved with **just 2 hours of on-robot training**.

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| `RLTEncoder` | `models/rlt_encoder.py` | Transformer compressing VLM hidden states → rl_token via cross-attn pooling |
| `RLTDecoder` | `models/rlt_encoder.py` | Reconstruction decoder for bottleneck pre-training |
| `RLTActor` | `models/actor.py` | SAC actor: `[rl_token, vla_action]` → tanh-squashed delta |
| `RLTCritic` | `models/critic.py` | Twin-Q SAC critic |
| `GR00TWrapperWithHooks` | `models/groot_wrapper.py` | Hooks last N VLM layers to expose hidden states |
| `SAC` | `training/sac.py` | SAC with reference regularization + auto-entropy tuning |
| `RLTTrainer` | `training/trainer.py` | Two-phase trainer: offline pre-train → online SAC |

## Quickstart

```bash
# 0. Install uv (fast Python package manager)
curl -Ls https://astral.sh/uv/install.sh | sh

# 1. Install GR00T N1
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T && uv pip install -e . && cd ..

# 2. Install groot-rlt
git clone https://github.com/akashspacesky/groot-rlt
cd groot-rlt
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Validate shapes (no GPU / GR00T needed)
python scripts/pretrain_rlt.py --dry_run --mock

# 4. Run unit tests
pytest tests/ -v

# 5. Phase 1: Pre-train bottleneck on demos
python scripts/pretrain_rlt.py \
    --config configs/default.yaml \
    --data_dir /path/to/lerobot_demos \
    --output checkpoints/bottleneck.pt

# 6. Phase 2: Online SAC on robot
python scripts/train.py \
    --config configs/tasks/ethernet_insertion.yaml \
    --encoder_ckpt checkpoints/bottleneck.pt

# 7. Evaluate
python scripts/evaluate.py \
    --config configs/tasks/ethernet_insertion.yaml \
    --checkpoint runs/rlt_ethernet_insertion/sac_latest.pt \
    --encoder_ckpt checkpoints/bottleneck.pt \
    --num_episodes 50 \
    --compare_baseline
```

## Key Hyperparameters

| Param | Default | Effect |
|-------|---------|--------|
| `rlt_dim` | 256 | RL token size. Smaller = stronger bottleneck pressure |
| `ref_reg_weight` | 0.1 | L2 penalty toward VLA action. Higher = stay closer to GR00T |
| `ref_dropout_prob` | 0.3 | Dropout on VLA reference fed to actor. Annealed during training |
| `updates_per_step` | 200 | SAC gradient steps per env step. High = fast on-robot convergence |
| `action_scale` | 0.1 | Max delta magnitude per step. Tune per task precision requirements |
| `num_hook_layers` | 4 | VLM layers to hook for hidden states |

## Adding a New Task

1. Add `configs/tasks/<task_name>.yaml` with task-specific overrides
2. Implement reward in `groot_rlt/envs/robot_env.py` → `_compute_reward`
3. Subclass `RobotEnv` with your robot interface (`_execute_action_chunk`)

## Development Notes

- **No GR00T required for development**: use `--mock` flag in all scripts, or `MockGR00TWrapper` in tests
- **Phase 1 is optional**: if you have no demos, skip to Phase 2 with random encoder init (works but slower convergence)
- **Human demos**: pass `is_human_demo=True` to `replay_buffer.add()` — these are never overwritten and sampled at `human_demo_ratio` in every batch

## Roadmap

- [ ] Dense reward from force-torque feedback
- [ ] Multi-task RLT: shared encoder, task-conditioned actor
- [ ] Sim2Real: pre-train in Isaac Lab, fine-tune on real robot
- [ ] Hook GR00T's DiT action head (not just VLM) for richer state
- [ ] Hierarchical: high-level VLA token + low-level RLT precision token

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
