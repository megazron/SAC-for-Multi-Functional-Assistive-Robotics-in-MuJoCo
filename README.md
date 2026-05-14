# 🤖 SAC for Multi-Functional Assistive Robotics in MuJoCo

> *Can a single reinforcement learning algorithm learn to assist across completely different robotic tasks?*

This project investigates whether a single, universal **Soft Actor-Critic (SAC)** implementation can robustly learn multiple distinct assistive functions across a variety of MuJoCo environments — without any environment-specific modifications to the core algorithm.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Results](#results)
- [License](#license)

---

## Overview

Assistive robotics demands agents that can generalise — performing tasks like balance assistance, manipulation, and locomotion support within a unified learning framework. Rather than designing task-specific policies, this project tests the hypothesis that SAC's off-policy, entropy-maximising approach is flexible enough to master diverse assistive control tasks using a single codebase.

Environments are sourced from the [Gymnasium](https://gymnasium.farama.org/) MuJoCo suite (e.g. `InvertedPendulum-v4`, `HalfCheetah-v4`, `Hopper-v4`, and others), providing physically realistic simulations of the kinds of dynamic challenges found in assistive robotics.

---

## Algorithm

**Soft Actor-Critic (SAC)** is a state-of-the-art off-policy deep reinforcement learning algorithm that optimises a maximum-entropy objective, encouraging both high reward *and* diverse behaviour. Key components in this implementation:

| Component | Detail |
|---|---|
| **Policy** | Gaussian policy with reparameterisation trick (tanh squashing) |
| **Critics** | Twin Q-networks (clipped double-Q) to reduce overestimation bias |
| **Entropy tuning** | Automatic temperature (`α`) adjustment via a learned `log_alpha` |
| **Replay buffer** | Uniform experience replay with configurable capacity |
| **Target network** | Soft update via Polyak averaging (τ) |

The update rule at each step minimises:
- Critic loss: MSE against Bellman targets with entropy bonus
- Actor loss: maximise `Q(s,a) − α · log π(a|s)`
- Temperature loss (if `automatic_entropy_tuning=True`): drives entropy toward target `−|A|`

---

## Project Structure

```
SAC-for-Multi-Functional-Assistive-Robotics-in-MuJoCo/
│
├── SAC.py              # Core SAC implementation (agent, networks, replay buffer)
├── train.py            # Training loop, logging, checkpointing, and visualisation
│
├── SAC_preTrained/     # Saved model checkpoints (.pth)
├── SAC_logs/           # CSV training logs (episode, timestep, reward)
├── SAC_figs/           # Generated plots (reward curves, histograms, heatmaps)
└── SAC_gifs/           # Rendered environment GIFs at each checkpoint
```

---

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium + MuJoCo
- NumPy
- Matplotlib
- Seaborn
- Pandas
- Pillow
- tqdm

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/megazron/SAC-for-Multi-Functional-Assistive-Robotics-in-MuJoCo.git
cd SAC-for-Multi-Functional-Assistive-Robotics-in-MuJoCo

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch numpy gymnasium[mujoco] matplotlib seaborn pandas pillow tqdm
```

> **Note:** MuJoCo requires a valid installation. Follow the [Gymnasium MuJoCo setup guide](https://gymnasium.farama.org/environments/mujoco/) if you encounter issues.

---

## Usage

### Training

To start training, simply run:

```bash
python train.py
```

Training will:
1. Initialise the SAC agent and the specified Gymnasium environment
2. Collect random exploration data for the first `start_steps` timesteps
3. Begin policy updates, logging rewards every `log_freq` steps
4. Save model checkpoints and generate GIFs + plots every `save_model_freq` steps
5. Produce a final GIF and full visualisation suite at the end of training

### Changing the Environment

Open `train.py` and modify the `env_name` variable near the top of the `train()` function:

```python
env_name = "InvertedPendulum-v4"   # ← change to any MuJoCo environment
```

Some suitable assistive environments to try:

| Environment | Task |
|---|---|
| `InvertedPendulum-v4` | Balance a pole on a cart |
| `InvertedDoublePendulum-v4` | Balance a two-link pendulum |
| `Hopper-v4` | One-legged locomotion |
| `HalfCheetah-v4` | Bipedal running |
| `Humanoid-v4` | Full humanoid locomotion |
| `Ant-v4` | Quadruped locomotion |

---

## Configuration

Key hyperparameters are set at the top of `train()` in `train.py`:

```python
# Environment
max_ep_len             = 1000       # Max steps per episode
max_training_timesteps = int(2.5e6) # Total environment steps

# SAC
lr_actor               = 0.0003
lr_critic              = 0.0003
gamma                  = 0.99       # Discount factor
tau                    = 0.005      # Polyak averaging coefficient
alpha                  = 0.2        # Initial entropy temperature
hidden_size            = 256        # Hidden layer width
replay_size            = int(1e6)   # Replay buffer capacity
batch_size             = 256
start_steps            = 5000       # Random exploration before learning

# Logging & checkpointing
log_freq               = 500
print_freq             = 5000
save_model_freq        = int(1e4)
```

`automatic_entropy_tuning = True` is strongly recommended — it removes the need to hand-tune `alpha` per environment.

---

## Outputs

After training, the following are automatically generated:

**`SAC_logs/`** — CSV files with columns `episode, timestep, reward` for post-hoc analysis.

**`SAC_figs/<env_name>/checkpoint_<t>/`** — Four plots per checkpoint:
- `reward_curve.png` — Raw reward over time
- `smoothed_reward.png` — Rolling-window smoothed reward
- `reward_histogram.png` — Distribution of episode rewards
- `reward_heatmap.png` — Episode-by-group reward heatmap

**`SAC_gifs/<env_name>/`** — Animated GIFs of the policy in action at each checkpoint, plus a `final.gif` of the fully-trained agent.

**`SAC_preTrained/<env_name>/`** — `.pth` checkpoint files containing policy, critic, target critic, and optimiser states.

---

## Results

Pre-trained checkpoints for select environments are included in `SAC_preTrained/`. To load and evaluate a saved model, you can use the `generate_gif_and_visuals` function directly from `train.py`, or load within your own script:

```python
from SAC import SAC
import gymnasium as gym

env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
agent = SAC(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    # ... hyperparameters ...
    action_space=env.action_space
)
agent.load("SAC_preTrained/InvertedPendulum-v4/SAC_InvertedPendulum-v4_0_0.pth")

state, _ = env.reset()
done = False
while not done:
    action = agent.select_action(state, evaluate=True)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <em>Built with PyTorch · Gymnasium · MuJoCo</em>
</p>
