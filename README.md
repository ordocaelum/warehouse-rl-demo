# Warehouse RL Demo

A reinforcement learning environment for warehouse logistics simulations built with **Gymnasium** and **Stable Baselines3**.

## Features

- **Custom Gymnasium Environment**: Configurable grid-based warehouse with dynamic package placement
- **DQN Agent**: Q-learning based deep neural network agent for autonomous decision-making
- **Reward Shaping**: Incentivizes efficient package delivery with step penalties and delivery bonuses
- **Interactive Training Lab**: Full PyGame UI (`ui_dashboard.py`) with live visualization and parameter controls
- **Live PyGame Visualizer**: Lightweight real-time visualization while training (`--visualize`)
- **Training Visualization**: Post-training performance charts with matplotlib
- **Testing & Inference**: Scripts for evaluating trained agent behavior

## Project Structure

```
warehouse-rl-demo/
├── warehouse_env.py       # Custom Gymnasium environment
├── train_agent.py         # Training script with DQN agent
├── ui_dashboard.py        # Interactive RL Training Lab (PyGame UI)
├── visualizer_pygame.py   # Lightweight live PyGame visualizer (threaded)
├── lab_render.py          # Renderer interface: how the lab draws an environment
├── lab_envs.py            # Observation as render state, for envs that report none
├── lab_metrics.py         # Algorithm metrics, forwarded by name
├── lab_params.py          # Declarative schema for the dashboard's settings
├── warehouse_render.py    # Renderer for WarehouseEnv's render state
├── warehouse_lab.py       # WarehouseEnv's settings for the dashboard
├── requirements.txt       # Python dependencies (pinned)
├── requirements-dev.txt   # + pytest
├── requirements.lock      # every package pinned, all platforms
├── tests/                 # pytest suite
├── runs/                  # one folder per training run (created on first run)
├── README.md              # This file
└── .gitignore             # Git configuration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ordocaelum/warehouse-rl-demo.git
cd warehouse-rl-demo
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.lock
```

`requirements.lock` pins every package, including transitive ones, for all
platforms; it is what the test suite was run with. `requirements.txt` pins
only the direct dependencies. Use Python 3.12 or 3.13: numpy 2.5 needs 3.12+,
and pygame 2.6.1 has no Python 3.14 wheels.

## Usage

### Interactive Training Lab (Recommended)

Launch the full-featured interactive UI:

```bash
python ui_dashboard.py
```

A 1300×780 window opens with three panels:

#### Left Panel – Parameter Controls

| Slider | Range | Default | Description |
|--------|-------|---------|-------------|
| Grid Size | 6 – 15 | 8 | Warehouse grid dimensions |
| Packages | 1 – 5 | 1 | Number of packages to deliver |
| Max Steps | 100 – 500 | 300 | Episode step limit |
| Step Cost | 0.01 – 0.5 | 0.05 | Per-step reward penalty |
| Learning Rate | 1e-4 – 1e-2 | 1e-3 | DQN learning rate (log scale) |
| Init Epsilon | 0.5 – 1.0 | 1.0 | Starting exploration rate |
| Final Epsilon | 0.01 – 0.3 | 0.1 | Ending exploration rate |
| Buffer (k) | 1k – 50k | 10k | Replay buffer size |
| Timesteps (k) | 10k – 500k | 100k | Total training timesteps |
| Seed | 0 – 999 | 0 | Random seed for the environment, agent and torch |

The sliders are generated from a declarative schema (`lab_params.py`): each
setting names the keyword argument it sets, whether it goes to the
environment, the agent or the run, its type, range, default, unit scale and
whether the slider is logarithmic. The environment's settings come first
(`warehouse_lab.py`), then the agent's and the run's.

#### Center Panel – Live Warehouse Grid

- Agent position updates in real time during training
- White circle = agent (empty-handed), gold circle = agent (carrying)
- Blue squares = packages pending delivery, green squares = delivered
- Green cell (D) = delivery zone
- Episode number, step count, and reward overlay below the grid

#### Right Panel – Stats & Metrics

- Training state, model status
- Live episode number, step, reward, and 50-episode moving average
- Algorithm metrics, whichever the agent logs, by name (for DQN: exploration
  rate, learning rate, number of updates, loss)
- Total timesteps, elapsed time, UI FPS
- Training progress bar
- Reward history graph (last 100 episodes) with 20-episode moving average

#### Bottom Bar – Control Buttons

| Button | Shortcut | Description |
|--------|----------|-------------|
| START TRAINING | — | Begin training with current slider settings |
| PAUSE / RESUME | **SPACE** | Pause or resume the training loop |
| STOP | — | Terminate training gracefully |
| SAVE MODEL | **S** | Save model and replay buffer to the current run's folder, `runs/<run>/model.zip`. Works while training or paused |
| LOAD MODEL | — | Select the most recent saved model and set the environment sliders to match; the next START resumes it (weights, optimizer, hyperparameters, step count, replay buffer) |
| RESET STATS | **R** | Clear reward history and metrics |
| EXIT | **Q** / **ESC** | Stop training and close the window |

### Training the Agent (Command Line)

Run the training script to train a DQN agent:

```bash
python train_agent.py
```

This will:
- Initialize a DQN agent with Q-learning principles
- Train for 100,000 timesteps
- Save the trained model as `runs/<run>/model.zip`
- Generate `runs/<run>/training_progress.png` showing reward curves
- Test the agent on 3 episodes, on the same environment configuration it trained on

Options: `--timesteps`, `--seed`, `--grid-size`, `--packages`, `--max-steps`,
`--step-cost`, `--checkpoint-freq`, `--runs-dir` (see `python train_agent.py --help`).

### Run Folders

Every training run, from the command line or the dashboard, writes to its own
folder, `runs/<YYYYmmdd-HHMMSS>_g<grid>_p<packages>_s<seed>/`, so runs never
overwrite each other:

| File | Contents |
|------|----------|
| `config.json` | Environment and agent settings, seed, timesteps, Python and library versions |
| `model.zip`, `model_replay_buffer.pkl` | The trained model (SB3 format) and its replay buffer |
| `checkpoints/` | Model and replay buffer every 10,000 steps (`--checkpoint-freq`) |
| `0.monitor.csv` | Reward, length and wall time of every episode |
| `progress.csv`, `events.out.tfevents.*` | SB3 training metrics; view with `tensorboard --logdir runs` |
| `train.log`, `training_progress.png` | Command-line runs only: console log and reward plot |

### Live Interactive Visualizer

Watch the agent learn in real-time with the PyGame visualizer:

```bash
python train_agent.py --visualize
```

A window opens showing:
- The 8×8 warehouse grid with agent, packages, and delivery zone
- Live metrics: episode, step, reward, 50-episode moving average, and the
  first metric the algorithm logs (for DQN, the exploration rate)
- A reward history sparkline

#### Visualizer Controls

| Key / Action         | Effect                                  |
|----------------------|-----------------------------------------|
| **SPACE**            | Pause / resume training                 |
| **S** (while paused) | Run to the end of the current episode, then pause on its final frame |
| **↑ / ↓**            | Zoom in / out                           |
| **Mouse wheel**      | Zoom in / out                           |
| **Mouse drag**       | Pan the grid                            |
| **R**                | Reset zoom and pan                      |

#### Visual Legend

| Element               | Colour                                  |
|-----------------------|-----------------------------------------|
| Agent (empty-handed)  | White circle                            |
| Agent (carrying)      | Gold / orange circle                    |
| Package (pending)     | Blue square                             |
| Package (delivered)   | Green square                            |
| Delivery zone (D)     | Green cell (bottom-right corner)        |
| Positive reward text  | Green                                   |
| Negative reward text  | Red                                     |

#### Threading Architecture

The visualizer's window runs on the **main thread** (pygame/SDL requires this
on macOS) and training runs in a **background thread**, so neither waits on the
other. Closing the window early lets training finish without display.  Training produces one state update
per step, far more than the display can draw, so updates go through a
keep-latest slot (`latest_slot.py`): training never waits, each frame draws the
newest state, and the panel shows how many steps were not drawn. Episode
returns use a separate channel, so none are lost.

### Customization

Edit parameters in `train_agent.py`:

```python
# Modify training duration
train_agent(total_timesteps=100000)

# Test on more episodes (pass the configuration the model was trained on)
test_agent("runs/<run>/model", num_episodes=5, grid_size=10, num_packages=2)
```

Edit environment parameters via constructor arguments in `warehouse_env.py` or `train_agent.py`:

```python
# Larger grid with more packages (once basic task is mastered)
env = WarehouseEnv(grid_size=10, num_packages=3)

# Adjust max steps and step cost
env = WarehouseEnv(max_steps=400, step_cost=0.02)

# Pass parameters to train_agent
train_agent(
    total_timesteps=200000,
    grid_size=10,
    num_packages=2,
    max_steps=400,
    learning_rate=5e-4,
)
```

## Environment Details

### Creating the Environment

```python
import gymnasium as gym
import warehouse_env  # registers "Warehouse-v0"

env = gym.make("Warehouse-v0", grid_size=8, num_packages=1, render_mode="ansi")
# or directly: warehouse_env.WarehouseEnv(grid_size=8, render_mode="human")
```

`render_mode="human"` prints the grid after every `reset()`/`step()`;
`render_mode="ansi"` makes `render()` return it as a string. The env truncates
episodes itself at `max_steps`, so no `TimeLimit` wrapper is registered.

`env.unwrapped.action_masks()` returns which of the six actions would change
the state (the convention sb3-contrib's `MaskablePPO` uses); the same mask is
in `info["action_mask"]`. DQN does not use it: masked actions remain legal
no-ops that cost the step cost.

### Action Space

| Action | Description |
|--------|-------------|
| 0 | Move up |
| 1 | Move down |
| 2 | Move left |
| 3 | Move right |
| 4 | Pickup package |
| 5 | Dropoff package |

### Observation Space

6-dimensional vector (for 1 package):
- Agent X, Y position (2 dims)
- Currently carrying package (1 dim, 0/1)
- Package 1: X, Y, delivery status (3 dims)

A package's X, Y is where the package currently is: its pickup cell while
waiting, the agent's cell while carried, and the delivery zone once delivered.

### Info Dictionary

`reset()` and `step()` return an `info` dict with:

| Key | Meaning |
|-----|---------|
| `delivered` | Packages delivered so far |
| `num_packages` | Packages in the episode |
| `invalid_action` | (`step` only) the action changed nothing: walking into a wall, picking up with nothing to pick up, dropping off empty-handed or away from the zone |
| `is_success` | (`step` only) every package has been delivered. SB3 reports it as `rollout/success_rate` |
| `termination_reason` | (`step` only) `"all_delivered"`, `"max_steps"` or `None`. A delivery on the last allowed step is `"all_delivered"` |
| `render_state` | Plain-data snapshot of what to draw (`get_render_state()`): `grid_size`, `agent_pos`, `carrying`, `packages` (`[x, y, delivered]` each), `delivery_zone` |

The dashboard and the visualizer draw only `render_state`. Their training
callbacks forward it from `info` and never read the environment object, so
they work unchanged with a `SubprocVecEnv`. On the last step of an episode it
still shows that step, although the vectorised env has already reset.

### Reward Structure

- **+10** for delivering a package to the delivery zone
- **+1** for picking up a package
- **+0.1** bonus for each step that reduces distance to nearest package
- **-0.01 × distance** per step to penalize being far from the nearest package
- **-0.05** per step (reduced from -0.1 for gentler exploration penalty)

### Grid Layout

```
. . . . . . . .
. 1 . . . . . .
. . . . . . . .
. . . A . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . D
```

- **A** = Agent (lowercase 'a' when carrying)
- **1-3** = Packages (numbers)
- **D** = Delivery zones
- **.** = Empty space

## Running the Tests

```bash
pip install -r requirements-dev.txt   # or requirements.lock, which includes pytest
pytest
```

The suite covers the environment contract (observation layout and bounds,
each reward term, termination and truncation, Gymnasium's `check_env`),
seeding and reproducibility, the training callbacks (final frame of each
episode, episode stepping, reward history), the dashboard's state machine
(driven headlessly), run folders, checkpointing, saving mid-run and resuming.
It trains only tiny models and runs in well under a minute.

## Reproducibility

Runs are seeded.  `python train_agent.py --seed N` (default `0`) and the
dashboard's **Seed** slider pass the seed to the environment
(`make_vec_env(seed=...)`) and to `DQN(seed=...)`, which seeds the action
space, `numpy`, Python's `random` and torch.  The environment draws all of its
randomness from `self.np_random`, so `env.reset(seed=N)` always produces the
same package layout.

What is covered by tests (`pytest`):

- identical seeds give identical resets and identical step-by-step rollouts;
- the environment never touches the process-global `np.random`;
- two short CPU training runs with the same seed finish with identical
  network weights and identical episode rewards.

What is **not** guaranteed:

- **GPU training.** Some CUDA kernels are nondeterministic; seeding does not
  make GPU runs bit-identical.  SB3 uses a GPU automatically when torch can
  see one (`device="auto"`); the determinism test above runs on the CPU.
- **Different installs.** Results depend on the installed library versions,
  including SB3's hyperparameter defaults. Install from `requirements.lock`
  to match the tested versions exactly.

## DQN Agent Parameters

The command line and the dashboard build the agent with the same function
(`make_dqn` in `train_agent.py`), so the same settings train the same way from
either. Settings without a slider or flag are fixed:

```python
DQN(
    learning_rate=1e-3,           # Learning rate
    buffer_size=10000,            # Replay buffer size
    learning_starts=500,          # Steps before training starts
    target_update_interval=500,   # Update target network every N steps
    exploration_fraction=0.2,     # Explore for 20% of training
    exploration_initial_eps=1.0,  # Start with 100% exploration
    exploration_final_eps=0.1,    # End with 10% exploration
)
```

## Performance Metrics

After training, you'll see:
- **Episode reward plot**: Total reward per episode
- **Moving average plot**: 50-episode rolling average (smooths noise)
- **Test episodes**: Live rendering of agent behavior

## Next Steps / Extensions

- Add obstacles/shelves to the grid
- Implement multi-agent scenarios
- Use image-based observations (CNN policy)
- Add time-based delivery bonuses
- Try other algorithms (PPO, A2C, SAC)
- Export trained models for production

## Requirements

- Python 3.12 or 3.13 (tested on 3.13.2)
- gymnasium 1.4.0 (Gym successor)
- stable-baselines3 2.9.0
- numpy 2.5.4
- matplotlib 3.11.2
- torch 2.14.1
- pygame 2.6.1 (for the dashboard and the live visualizer)
- tensorboard 2.21.0

Exact versions of everything else are in `requirements.lock`.

## License

MIT License - feel free to use and modify for your projects.

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Deep Q-Networks (DQN)](https://arxiv.org/abs/1312.5602)
