# Warehouse RL Demo

A reinforcement learning environment for warehouse logistics simulations built with **Gymnasium** and **Stable Baselines3**.

## Features

- **Custom Gymnasium Environment**: 8x8 grid-based warehouse with dynamic package placement
- **DQN Agent**: Q-learning based deep neural network agent for autonomous decision-making
- **Reward Shaping**: Incentivizes efficient package delivery with step penalties and delivery bonuses
- **Live PyGame Visualizer**: Interactive real-time visualization while training (`--visualize`)
- **Training Visualization**: Post-training performance charts with matplotlib
- **Testing & Inference**: Scripts for evaluating trained agent behavior

## Project Structure

```
warehouse-rl-demo/
├── warehouse_env.py       # Custom Gymnasium environment
├── train_agent.py         # Training script with DQN agent
├── visualizer_pygame.py   # Live PyGame visualizer (threaded)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git configuration
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
pip install -r requirements.txt
```

## Usage

### Training the Agent

Run the training script to train a DQN agent:

```bash
python train_agent.py
```

This will:
- Initialize a DQN agent with Q-learning principles
- Train for 100,000 timesteps
- Save the trained model as `warehouse_dqn_model.zip`
- Generate `training_progress.png` showing reward curves
- Test the agent on 3 episodes

### Live Interactive Visualizer

Watch the agent learn in real-time with the PyGame visualizer:

```bash
python train_agent.py --visualize
```

A window opens showing:
- The 8×8 warehouse grid with agent, packages, and delivery zone
- Live metrics: episode, step, reward, 50-episode moving average, exploration rate
- A reward history sparkline

#### Visualizer Controls

| Key / Action         | Effect                                  |
|----------------------|-----------------------------------------|
| **SPACE**            | Pause / resume training                 |
| **S** (while paused) | Step through one episode                |
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

The visualizer runs in a **background daemon thread** so training continues on
the main CPU thread without interruption.  State updates are exchanged via a
thread-safe `queue.Queue`; the visualizer drops frames when it falls behind so
it never slows training down.

### Customization

Edit parameters in `train_agent.py`:

```python
# Modify training duration
train_agent(total_timesteps=100000)

# Test on more episodes
test_agent(num_episodes=5)
```

Edit environment parameters in `warehouse_env.py`:

```python
# Larger grid with more packages (once basic task is mastered)
env = WarehouseEnv(grid_size=10, num_packages=3)

# Adjust max steps per episode
self.max_steps = 400

# Modify reward values
reward += 15  # Higher delivery reward
```

## Environment Details

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

## DQN Agent Parameters

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

- Python 3.8+
- gymnasium (Gym successor)
- stable-baselines3
- numpy
- matplotlib
- torch (installed with stable-baselines3)
- pygame >= 2.1.0 (for the live visualizer)

## License

MIT License - feel free to use and modify for your projects.

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Deep Q-Networks (DQN)](https://arxiv.org/abs/1312.5602)
