import math
import numbers

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any


def _require_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


class WarehouseEnv(gym.Env):
    """
    Simplified warehouse grid environment for easier learning.
    - Grid size: 8x8 (smaller search space)
    - Single package delivery (focus on core task)
    - Better reward shaping with distance-based guidance
    - Reward: +10 for delivery, +1 for pickup, -0.05 per step
    """
    
    # "human" prints the grid after every reset/step; "ansi" returns it from render().
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(self, grid_size: int = 8, num_packages: int = 1,
                 max_steps: int = 300, step_cost: float = 0.05,
                 render_mode: str | None = None):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']} or None, "
                f"got {render_mode!r}"
            )
        self.render_mode = render_mode
        # Packages are placed strictly inside the border, which needs a grid
        # of at least 3x3; with no packages an episode would end immediately.
        grid_size = _require_int("grid_size", grid_size, 3)
        num_packages = _require_int("num_packages", num_packages, 1)
        max_steps = _require_int("max_steps", max_steps, 1)
        if not isinstance(step_cost, numbers.Real) or not math.isfinite(step_cost):
            raise ValueError(f"step_cost must be a finite number, got {step_cost!r}")
        self.grid_size = grid_size
        self.num_packages = num_packages
        
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=pickup, 5=dropoff
        self.action_space = spaces.Discrete(6)
        
        # State space: [agent_x, agent_y, carrying, package1_x, package1_y, package1_delivered, ...]
        # For 1 package: agent_pos(2) + carrying(1) + packages(1*(2+1)) = 6 dims
        # Bounds are exact: coordinates lie in [0, grid_size - 1], flags in {0, 1}.
        last = grid_size - 1
        high = np.array([last, last, 1] + [last, last, 1] * num_packages, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.zeros_like(high), high=high, dtype=np.float32
        )

        # Internal state is integer grid coordinates; only the observation is
        # float32.
        self.agent_pos = np.array([0, 0], dtype=np.int64)
        self.carrying = -1  # -1 = nothing, else package index
        self.packages = []  # List of [x, y, delivered]
        self.spawn_zone = (0, 0)
        self.delivery_zone = (grid_size - 1, grid_size - 1)
        
        self._init_packages()
        self.step_count = 0
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.last_distance = self._get_nearest_package_distance()  # For distance-based reward shaping
    
    def _init_packages(self):
        """Randomly place packages in the grid."""
        self.packages = []
        # Use the env's own generator (seeded by reset(seed=...)), never the
        # process-global np.random, so identical seeds give identical layouts.
        for i in range(self.num_packages):
            x = int(self.np_random.integers(1, self.grid_size - 1))
            y = int(self.np_random.integers(1, self.grid_size - 1))
            self.packages.append([x, y, 0])  # [x, y, delivered]
        self.last_distance = float('inf')

    def _get_nearest_package_distance(self) -> float:
        """Calculate Manhattan distance to nearest undelivered package."""
        undelivered = [pkg for pkg in self.packages if not pkg[2]]
        if not undelivered:
            return 0.0
        return float(min(
            abs(self.agent_pos[0] - pkg[0]) + abs(self.agent_pos[1] - pkg[1])
            for pkg in undelivered
        ))

    def action_masks(self) -> np.ndarray:
        """Which of the 6 actions would change the state right now.

        Same convention as sb3-contrib's MaskablePPO.  Masked actions are
        still allowed; they cost the step cost and do nothing.
        """
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        last = self.grid_size - 1
        can_pickup = self.carrying == -1 and any(
            p[0] == x and p[1] == y and not p[2] for p in self.packages
        )
        can_dropoff = self.carrying >= 0 and (x, y) == tuple(self.delivery_zone)
        return np.array(
            [y > 0, y < last, x > 0, x < last, can_pickup, can_dropoff], dtype=bool
        )

    def get_render_state(self) -> Dict[str, Any]:
        """Everything needed to draw the current frame, as plain Python values.

        A fresh snapshot that shares nothing with the env.  reset() and step()
        report it as ``info["render_state"]``, so displays draw what the env
        pushed instead of reading its attributes.  On the last step of an
        episode it still describes that step after a vectorised env has
        auto-reset.
        """
        return {
            "grid_size": self.grid_size,
            "agent_pos": [int(self.agent_pos[0]), int(self.agent_pos[1])],
            "carrying": int(self.carrying),
            "packages": [[int(x), int(y), int(d)] for x, y, d in self.packages],
            "delivery_zone": [int(c) for c in self.delivery_zone],
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "delivered": sum(1 for pkg in self.packages if pkg[2]),
            "num_packages": self.num_packages,
            "delivery_zone": [int(c) for c in self.delivery_zone],
            "action_mask": self.action_masks(),
            "render_state": self.get_render_state(),
        }

    def _get_obs(self) -> np.ndarray:
        """Flatten state into observation vector."""
        obs = [self.agent_pos[0], self.agent_pos[1], float(self.carrying >= 0)]
        for pkg in self.packages:
            obs.extend([pkg[0], pkg[1], float(pkg[2])])
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int64)
        self.carrying = -1
        self._init_packages()
        self.step_count = 0
        self.last_distance = self._get_nearest_package_distance()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step with improved reward shaping.
        Returns: observation, reward, terminated, truncated, info

        info keys:
          delivered          packages delivered so far
          num_packages       packages in the episode
          delivery_zone      [x, y] of the delivery zone
          action_mask        action_masks() for the *new* state
          render_state       get_render_state() for the *new* state
          invalid_action     True if the action changed nothing (walking into
                             a wall, pickup with nothing to pick up, dropoff
                             while empty-handed or away from the zone)
          is_success         True when every package has been delivered
          termination_reason "all_delivered", "max_steps" or None; a delivery
                             on the last allowed step reports "all_delivered"
        """
        a = int(action)
        invalid_action = not (0 <= a < 6 and self.action_masks()[a])
        reward = -self.step_cost  # Configurable step cost
        
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

        # A carried package moves with the agent, so a delivered one ends up
        # at the delivery zone where it was dropped.
        if self.carrying >= 0:
            pkg = self.packages[self.carrying]
            pkg[0], pkg[1] = int(self.agent_pos[0]), int(self.agent_pos[1])

        # Distance-based reward shaping: only during exploration (not carrying)
        if self.carrying == -1 and action in (0, 1, 2, 3):
            current_distance = self._get_nearest_package_distance()
            if current_distance < self.last_distance:
                reward += 0.1  # Bonus for moving closer to package
            reward -= 0.01 * current_distance  # Penalize proportionally to distance
            self.last_distance = current_distance
        
        self.step_count += 1
        terminated = all(pkg[2] for pkg in self.packages)  # All packages delivered
        truncated = self.step_count >= self.max_steps

        info = self._get_info()
        info["invalid_action"] = invalid_action
        info["is_success"] = terminated
        info["termination_reason"] = (
            "all_delivered" if terminated else "max_steps" if truncated else None
        )
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info
    
    def render(self):
        """Text rendering: printed in "human" mode, returned as a str in "ansi"."""
        if self.render_mode is None:
            gym.logger.warn(
                "render() called without a render_mode; pass "
                "render_mode='human' or 'ansi' to WarehouseEnv(...)."
            )
            return None

        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark delivery zone
        grid[self.delivery_zone[1]][self.delivery_zone[0]] = 'D'
        
        # Mark packages
        for i, pkg in enumerate(self.packages):
            if not pkg[2]:
                grid[pkg[1]][pkg[0]] = str(i + 1)
        
        # Mark agent
        grid[self.agent_pos[1]][self.agent_pos[0]] = 'A' if self.carrying == -1 else 'a'
        
        lines = [''.join(row) for row in grid]
        lines.append(
            f"Step: {self.step_count}, Carrying: {self.carrying}, "
            f"Delivered: {sum(p[2] for p in self.packages)}\n"
        )
        text = "\n".join(lines)
        if self.render_mode == "ansi":
            return text
        print(text)
        return None


# Registered so that gym.make("Warehouse-v0", grid_size=..., render_mode=...)
# works once this module is imported.  No max_episode_steps: the env truncates
# itself at max_steps.
ENV_ID = "Warehouse-v0"
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="warehouse_env:WarehouseEnv")
