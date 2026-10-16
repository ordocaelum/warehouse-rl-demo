"""Declarative schema for the training lab's tunable settings.

Each :class:`Param` names the keyword argument it sets and what receives it:

- ``"env"``: the environment constructor (``gym.make(env_id, **env_kwargs)``)
- ``"agent"``: the agent factory (``make_dqn(env, **agent_kwargs)``)
- ``"run"``: the training run itself (``total_timesteps``, ``seed``)

The dashboard builds one slider per Param and passes values on by name, so
adding, removing or reordering a setting is a change to a list, not to code
that indexes sliders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

TARGETS = ("env", "agent", "run")


@dataclass(frozen=True)
class Param:
    name: str                  # keyword argument it sets
    target: str                # one of TARGETS
    type: type                 # int or float
    min: float                 # slider range and default, in shown units
    max: float
    default: float
    scale: float = 1           # value passed on = shown value * scale
    log: bool = False          # logarithmic slider
    label: str = ""            # slider label (defaults to name)
    fmt: Optional[str] = None  # format of the shown value, e.g. "{:.2e}"

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"{self.name}: target must be one of {TARGETS}, got {self.target!r}")
        if self.type not in (int, float):
            raise ValueError(f"{self.name}: type must be int or float, got {self.type!r}")
        if not self.min < self.max:
            raise ValueError(f"{self.name}: min must be below max")
        if not self.min <= self.default <= self.max:
            raise ValueError(f"{self.name}: default {self.default} is outside [{self.min}, {self.max}]")
        if self.log and self.min <= 0:
            raise ValueError(f"{self.name}: a log-scale range must be positive")
        if not self.label:
            object.__setattr__(self, "label", self.name)

    def value(self, shown: float):
        """The value passed on, for a slider showing ``shown``."""
        v = shown * self.scale
        return int(round(v)) if self.type is int else float(v)

    def shown(self, value: float) -> float:
        """The slider position, in shown units, for a passed-on ``value``."""
        return value / self.scale

    def in_range(self, value: float) -> bool:
        shown = self.shown(value)
        return math.isfinite(shown) and self.min <= shown <= self.max


def by_target(params: Iterable[Param], values: Mapping[str, object]) -> Dict[str, dict]:
    """Group ``values`` (passed-on values keyed by Param name) by target.

    Returns ``{"env": {...}, "agent": {...}, "run": {...}}``. Every Param
    must have a value; names that are not Params are an error.
    """
    params = list(params)
    unknown = set(values) - {p.name for p in params}
    if unknown:
        raise KeyError(f"not parameters: {sorted(unknown)}")
    grouped: Dict[str, dict] = {t: {} for t in TARGETS}
    for p in params:
        grouped[p.target][p.name] = values[p.name]
    return grouped


# DQN settings with a slider (make_dqn in train_agent.py takes these).
AGENT_PARAMS = (
    Param("learning_rate", "agent", float, 1e-4, 1e-2, 1e-3, log=True,
          label="Learning Rate", fmt="{:.2e}"),
    Param("exploration_initial", "agent", float, 0.5, 1.0, 1.0, label="Init Epsilon"),
    Param("exploration_final", "agent", float, 0.01, 0.3, 0.1, label="Final Epsilon"),
    Param("buffer_size", "agent", int, 1, 50, 10, scale=1000, label="Buffer (k)"),
)

# Settings of the training run, whatever the environment and agent.
RUN_PARAMS = (
    Param("total_timesteps", "run", int, 10, 500, 100, scale=1000, label="Timesteps (k)"),
    Param("seed", "run", int, 0, 999, 0, label="Seed"),
)
