"""Environment-side support the training lab adds to every environment it trains."""

from __future__ import annotations

import math

import gymnasium as gym
from gymnasium import spaces


class ObservationRenderState(gym.Wrapper):
    """Reports the observation as ``info["render_state"]`` for envs that report none.

    Envs that push their own render state (WarehouseEnv) pass through
    unchanged. For any other env the state is
    ``{"observation": [...], "low": [...], "high": [...]}``: the flattened
    observation and its bounds, with None for an unbounded component.
    lab_render.ObservationRenderer draws it.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        flat = spaces.flatten_space(env.observation_space)

        def bounds(values):
            return [float(v) if math.isfinite(v) else None for v in values.ravel()]

        self._low, self._high = bounds(flat.low), bounds(flat.high)

    def _with_render_state(self, obs, info: dict) -> dict:
        if "render_state" in info:
            return info
        flat = spaces.flatten(self.env.observation_space, obs)
        return {**info, "render_state": {
            "observation": [float(v) for v in flat],
            "low": list(self._low),
            "high": list(self._high),
        }}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._with_render_state(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._with_render_state(obs, info)
