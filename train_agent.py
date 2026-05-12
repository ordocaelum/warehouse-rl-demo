import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from warehouse_env import WarehouseEnv


class RewardCallback(BaseCallback):
    """Track episode rewards during training, with optional live visualizer support."""

    def __init__(self, visualizer=None, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0
        self._visualizer = visualizer
        self._episode_num = 0
        self._step_in_ep = 0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        self.current_reward += reward
        self._step_in_ep += 1

        # Push live update to visualizer (non-blocking)
        if self._visualizer is not None:
            env = self.training_env.envs[0]
            # Unwrap Monitor/VecEnv layers to reach the raw WarehouseEnv
            raw_env = env
            while hasattr(raw_env, "env"):
                raw_env = raw_env.env

            avg_reward = None
            window = self.episode_rewards[-50:]
            if len(self.episode_rewards) >= 50:
                avg_reward = float(np.mean(window))

            epsilon = 1.0
            if hasattr(self.model, "exploration_rate"):
                epsilon = float(self.model.exploration_rate)

            self._visualizer.put_update({
                "episode":   self._episode_num,
                "step":      self._step_in_ep,
                "reward":    float(reward),
                "ep_reward": float(self.current_reward),
                "avg_reward": avg_reward,
                "epsilon":   epsilon,
                "agent_pos": raw_env.agent_pos.tolist(),
                "packages":  [list(p) for p in raw_env.packages],
                "carrying":  int(raw_env.carrying),
                "grid_size": int(raw_env.grid_size),
            })

            # Honour pause/resume from visualizer; bail out if it stops running
            while self._visualizer.is_paused() and self._visualizer.is_running():
                if self._visualizer.consume_step_request():
                    break  # step one full episode
                time.sleep(0.05)

        # End of episode bookkeeping
        if self.locals.get("dones", [False])[0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
            self._episode_num += 1
            self._step_in_ep = 0

        return True


def train_agent(total_timesteps: int = 100000, visualizer=None):
    """Train DQN agent on warehouse environment."""

    # Create environment
    env = make_vec_env(WarehouseEnv, n_envs=1)

    # Initialize DQN agent
    # Note: DQN is based on Q-learning principles
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=500,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        verbose=1
    )

    # Train with callback
    callback = RewardCallback(visualizer=visualizer)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    model.save("warehouse_dqn_model")

    return model, callback, env

def visualize_training(callback: RewardCallback):
    """Plot training progress."""
    plt.figure(figsize=(12, 5))
    
    # Episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(callback.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress: Episode Rewards")
    plt.grid(True)
    
    # Moving average
    plt.subplot(1, 2, 2)
    window = 50
    if len(callback.episode_rewards) > window:
        moving_avg = np.convolve(callback.episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward (50-ep window)")
        plt.title("Training Progress: Moving Average")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.show()

def test_agent(model_path: str = "warehouse_dqn_model", num_episodes: int = 3):
    """Test trained agent."""
    env = WarehouseEnv()
    model = DQN.load(model_path)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()
        
        print(f"Episode {episode + 1} Total Reward: {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train warehouse DQN agent")
    parser.add_argument(
        "--visualize", action="store_true",
        help="Open a live PyGame visualizer while training"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000,
        help="Total training timesteps (default: 100000)"
    )
    args = parser.parse_args()

    visualizer = None
    if args.visualize:
        from visualizer_pygame import WarehouseVisualizer
        visualizer = WarehouseVisualizer(grid_size=8)
        visualizer.start()
        print("Live visualizer started. Controls: SPACE=pause  S=step  ↑↓=zoom  drag=pan  R=reset")

    # Train agent
    print("Training agent...")
    model, callback, env = train_agent(
        total_timesteps=args.timesteps,
        visualizer=visualizer,
    )

    if visualizer is not None:
        visualizer.stop()

    # Visualize
    print("Plotting training progress...")
    visualize_training(callback)

    # Test
    print("Testing trained agent...")
    test_agent(num_episodes=3)
