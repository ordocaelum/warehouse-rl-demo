"""How the training lab handles environments: any Gymnasium env id, plus
optional per-environment profiles (sliders, renderer, run-folder name)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

from lab_params import Param
from lab_render import ObservationRenderer, Renderer


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


@dataclass(frozen=True)
class EnvProfile:
    """What the lab knows about one environment id, beyond Gymnasium's registry.

    Nothing is required: an id without a registered profile trains with no
    environment sliders, is drawn from its observation, and names its run
    folders after the id.
    """

    env_id: str
    params: Tuple[Param, ...] = ()                          # env constructor sliders
    renderer: Callable[[], Renderer] = ObservationRenderer  # renderer factory
    run_name: Optional[Callable[[dict], str]] = None        # run-folder tag from env kwargs

    def run_tag(self, env_kwargs: dict) -> str:
        if self.run_name is not None:
            return self.run_name(env_kwargs)
        return re.sub(r"[^A-Za-z0-9.-]+", "-", self.env_id)


_PROFILES: Dict[str, EnvProfile] = {}


def register_profile(profile: EnvProfile) -> None:
    _PROFILES[profile.env_id] = profile


def get_profile(env_id: str) -> EnvProfile:
    """The registered profile for ``env_id``, or the default one."""
    return _PROFILES.get(env_id) or EnvProfile(env_id)
