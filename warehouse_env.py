import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

class WarehouseEnv(gym.Env):
    """
    Simple warehouse grid environment.
    - Grid size: 10x10
    - Agent moves packages from spawn to delivery zones
    - Reward: +10 for delivery, -0.1 per step (encourages efficiency)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, grid_size: int = 10, num_packages: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.num_packages = num_packages
        
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=pickup, 5=dropoff
        self.action_space = spaces.Discrete(6)
        
        # State space: [agent_x, agent_y, carrying, package1_x, package1_y, package1_delivered, ...]
        # For 3 packages: agent_pos(2) + carrying(1) + packages(3*(2+1)) = 14 dims
        state_dim = 2 + 1 + (num_packages * 3)
        self.observation_space = spaces.Box(
            low=0, 
            high=max(grid_size, 1),  # Handle package_delivered as 0/1
            shape=(state_dim,),
            dtype=np.float32
        )
        
        self.agent_pos = np.array([0, 0])
        self.carrying = -1  # -1 = nothing, else package index
        self.packages = []  # List of [x, y, delivered]
        self.spawn_zone = (0, 0)
        self.delivery_zone = (grid_size - 1, grid_size - 1)
        
        self._init_packages()
        self.step_count = 0
        self.max_steps = 200
    
    def _init_packages(self):
        """Randomly place packages in the grid."""
        self.packages = []
        for i in range(self.num_packages):
            x = np.random.randint(1, self.grid_size - 1)
            y = np.random.randint(1, self.grid_size - 1)
            self.packages.append([x, y, 0])  # [x, y, delivered]
    
    def _get_obs(self) -> np.ndarray:
        """Flatten state into observation vector."""
        obs = [self.agent_pos[0], self.agent_pos[1], float(self.carrying >= 0)]
        for pkg in self.packages:
            obs.extend([pkg[0], pkg[1], float(pkg[2])])
        return np.array(obs, dtype=np.float32)
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.carrying = -1
        self._init_packages()
        self.step_count = 0
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.
        Returns: observation, reward, terminated, truncated, info
        """
        reward = -0.1  # Step cost
        
        # Movement actions
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        
        # Pickup action
        elif action == 4:
            if self.carrying == -1:  # Not carrying
                for i, pkg in enumerate(self.packages):
                    if (self.agent_pos == pkg[:2]).all() and not pkg[2]:
                        self.carrying = i
                        reward += 1  # Pickup reward
                        break
        
        # Dropoff action
        elif action == 5:
            if self.carrying >= 0:
                pkg = self.packages[self.carrying]
                if (self.agent_pos == list(self.delivery_zone)).all():
                    pkg[2] = 1  # Mark delivered
                    reward += 10  # Delivery reward
                    self.carrying = -1
        
        self.step_count += 1
        terminated = all(pkg[2] for pkg in self.packages)  # All packages delivered
        truncated = self.step_count >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def render(self):
        """Simple text rendering."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark delivery zone
        grid[self.delivery_zone[1]][self.delivery_zone[0]] = 'D'
        
        # Mark packages
        for i, pkg in enumerate(self.packages):
            if not pkg[2]:
                grid[pkg[1]][pkg[0]] = str(i + 1)
        
        # Mark agent
        grid[self.agent_pos[1]][self.agent_pos[0]] = 'A' if self.carrying == -1 else 'a'
        
        for row in grid:
            print(''.join(row))
        print(f"Step: {self.step_count}, Carrying: {self.carrying}, Delivered: {sum(p[2] for p in self.packages)}\n")
