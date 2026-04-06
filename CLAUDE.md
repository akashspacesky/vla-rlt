# groot-rlt

## Project Overview

**groot-rlt** implements Physical Intelligence's RL Token (RLT) idea on NVIDIA's GR00T N1 open-source VLA. The core thesis: pre-trained VLAs are good at general manipulation but lack the precision needed for tight-tolerance tasks (ethernet insertion, screwdriver alignment, zip-tie fastening). RLT adds a compact bottleneck representation extracted from the VLA's internal embeddings, then trains a small actor-critic on that token using online RL — achieving ~3x speedup on precision stages with just 15 minutes of real-world data.

**Paper**: [Physical Intelligence RLT](https://www.pi.website/research/rlt)  
**Base model**: [NVIDIA Isaac GR00T N1](https://github.com/NVIDIA/Isaac-GR00T)

---

## Architecture

```
Observation (images + language)
        │
   [GR00T N1 VLM]  ← frozen during RL phase
        │
   internal hidden states (e.g., last 4 layers)
        │
   [RLT Encoder] ──bottleneck──► rl_token (d=256)
        │                              │
   [RLT Decoder]              [Actor MLP]   [Critic MLP]
   (reconstruction loss)      ↑                 ↑
                         vla_action         rl_token
                              │
                         delta_action  ← small learned edit
                              │
                    final_action = vla_action + delta_action
```

### Key Components

- **`groot_rlt/models/rlt_encoder.py`**: Encoder-decoder transformer. Compresses GR00T's VLM hidden states through an information bottleneck into a single `rl_token` vector. The decoder reconstructs VLM embeddings from this token (bottleneck pre-training loss).
- **`groot_rlt/models/actor.py`**: MLP actor. Takes `[rl_token, vla_action_chunk]` → `delta_action`. Edits rather than replaces VLA output.
- **`groot_rlt/models/critic.py`**: Twin-Q critic for SAC. Takes `[rl_token, action_chunk]` → Q-value.
- **`groot_rlt/models/groot_wrapper.py`**: Hooks into GR00T N1 to expose internal VLM hidden states alongside predicted actions.
- **`groot_rlt/training/sac.py`**: SAC with action chunking, reference regularization (KL penalty toward VLA action), reference-action dropout.
- **`groot_rlt/training/replay_buffer.py`**: Off-policy replay buffer storing `(rl_token, vla_action, delta, reward, next_rl_token, done)`.
- **`groot_rlt/training/trainer.py`**: Main training loop. Pre-trains encoder-decoder offline, then runs online SAC.

### Training Phases

1. **Phase 1 — Offline Bottleneck Pre-training** (`scripts/pretrain_rlt.py`):  
   Collect demonstrations, extract GR00T hidden states, train encoder-decoder with reconstruction loss. Encoder weights are frozen after this.

2. **Phase 2 — Online RL Fine-tuning** (`scripts/train.py`):  
   Run SAC on the robot. Only actor and critic are updated. GR00T and encoder stay frozen.

### Key Design Choices (from Pi paper)

- **Action chunking**: Actor predicts a chunk of T actions, matching GR00T's temporal structure.
- **Reference regularization**: SAC objective includes a KL/L2 penalty keeping actor close to VLA reference action. Prevents catastrophic forgetting of base policy.
- **Reference-action dropout**: During early training, randomly drop the VLA reference action fed to actor. Forces actor to learn independent policy pathway.
- **Human intervention integration**: Replay buffer can accept human-corrected trajectories as high-reward demonstrations.

---

## Repo Structure

```
groot-rlt/
├── CLAUDE.md                     ← you are here
├── README.md
├── pyproject.toml
├── groot_rlt/
│   ├── models/
│   │   ├── rlt_encoder.py        # Encoder-decoder bottleneck transformer
│   │   ├── actor.py              # SAC actor (action editing MLP)
│   │   ├── critic.py             # Twin-Q SAC critic
│   │   └── groot_wrapper.py      # GR00T N1 with hidden state hooks
│   ├── training/
│   │   ├── sac.py                # SAC algorithm implementation
│   │   ├── replay_buffer.py      # Off-policy replay buffer
│   │   └── trainer.py            # Orchestrates pre-training + RL loop
│   ├── envs/
│   │   └── robot_env.py          # Gym-compatible robot environment wrapper
│   └── utils/
│       ├── logging.py            # WandB + console logging
│       └── checkpointing.py      # Save/load model states
├── configs/
│   ├── default.yaml              # Base hyperparameters
│   └── tasks/
│       ├── ethernet_insertion.yaml
│       └── screwdriver_alignment.yaml
├── scripts/
│   ├── pretrain_rlt.py           # Phase 1: offline bottleneck pre-training
│   ├── train.py                  # Phase 2: online SAC
│   └── evaluate.py               # Evaluate trained policy
└── tests/
    ├── test_models.py
    └── test_training.py
```

---

## Development Guidelines

### Running Experiments

```bash
# Phase 1: pre-train the RLT encoder-decoder on offline demos
python scripts/pretrain_rlt.py --config configs/default.yaml --data_dir /path/to/demos

# Phase 2: online RL on robot
python scripts/train.py --config configs/tasks/ethernet_insertion.yaml --checkpoint rlt_pretrained.pt

# Evaluate
python scripts/evaluate.py --checkpoint runs/exp_name/best.pt --num_episodes 50
```

### Key Hyperparameters (configs/default.yaml)

- `rlt_dim: 256` — RL token dimensionality. Smaller = more bottleneck pressure.
- `chunk_size: 16` — Action chunk length, must match GR00T's.
- `ref_reg_weight: 0.1` — KL penalty weight toward VLA reference action.
- `ref_dropout_prob: 0.3` — Probability of dropping VLA reference during actor forward pass.
- `sac_alpha: 0.2` — SAC entropy temperature.
- `updates_per_step: 200` — How many gradient steps per environment step (key for fast on-robot training).

### What to Iterate On Next

- **Reward shaping**: Current impl uses sparse reward (success/failure). Dense rewards (e.g., distance to goal, force feedback) will help.
- **RLT token dimensionality**: Try 64, 128, 256, 512 and ablate.
- **Which layers to hook**: Default is last 4 VLM layers. Try hooking DiT action head layers too.
- **Multi-task RLT**: Single encoder shared across tasks, task-conditioned actor.
- **Real2Sim2Real**: Pre-train RLT in sim (Isaac Lab), transfer to real robot.
- **Hierarchical RLT**: High-level VLA token for task planning, low-level RLT for precision.

### Adding a New Task

1. Add a task config in `configs/tasks/<task_name>.yaml` with reward function path.
2. Implement the reward function in `groot_rlt/envs/robot_env.py` or a task-specific file.
3. Register the embodiment in GR00T's config if using a new robot.

---

## Environment Setup

```bash
# Clone GR00T N1
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T && pip install -e .

# Install groot-rlt
cd ../groot-rlt
pip install -e ".[dev]"
```

Requires: Python 3.10+, PyTorch 2.1+, CUDA 12.1+, GR00T N1 installed.

---

## Notes for Claude

- GR00T N1's VLM is based on Cosmos-Reason-2B. Hidden states are accessible via `output_hidden_states=True` in the forward pass.
- The DiT action head operates on a separate latent. For now we hook the VLM (language+vision side), not the DiT.
- SAC is preferred over PPO here because it is off-policy and dramatically more sample-efficient for real-robot settings.
- The actor outputs a *delta* on the action chunk, not a full replacement. This is critical — it ensures the VLA's learned behaviors are preserved and RL only refines.
- Reference-action dropout should be annealed: start high (0.5) and decay to 0 over first 1000 steps, then back up to 0.3.
