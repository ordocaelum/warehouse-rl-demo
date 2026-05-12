import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from warehouse_env import WarehouseEnv

class RewardCallback(BaseCallback):
    """Track episode rewards during training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0
    
    def _on_step(self) -> bool:
        # Get reward from info
        self.current_reward += self.locals.get("rewards", [0])[0]
        
        # Check if episode done
        if self.locals.get("dones", [False])[0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
        
        return True

def train_agent(total_timesteps: int = 50000):
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
        learning_starts=1000,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1
    )
    
    # Train with callback
    callback = RewardCallback()
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
    # Train agent
    print("Training agent...")
    model, callback, env = train_agent(total_timesteps=50000)
    
    # Visualize
    print("Plotting training progress...")
    visualize_training(callback)
    
    # Test
    print("Testing trained agent...")
    test_agent(num_episodes=3)
