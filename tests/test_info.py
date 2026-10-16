"""The info dict reports progress, invalid actions and why an episode ended."""

import numpy as np
import pytest

from helpers import DOWN, DROPOFF, LEFT, PICKUP, RIGHT, UP, deliver, pick_up, walk_to
from warehouse_env import WarehouseEnv


def _env(**kwargs):
    env = WarehouseEnv(grid_size=6, num_packages=2, **kwargs)
    env.reset(seed=0)
    env.packages = [[2, 2, 0], [3, 4, 0]]
    return env


def test_reset_info():
    _, info = WarehouseEnv(num_packages=3).reset(seed=0)
    assert info["delivered"] == 0
    assert info["num_packages"] == 3


def test_step_info_keys():
    env = _env()
    *_, info = env.step(RIGHT)
    assert {"delivered", "num_packages", "invalid_action",
            "is_success", "termination_reason"} <= info.keys()


@pytest.mark.parametrize("action", [UP, LEFT])
def test_walking_into_a_wall_is_invalid(action):
    env = _env()  # agent starts at (0, 0)
    *_, info = env.step(action)
    assert info["invalid_action"]


@pytest.mark.parametrize("action", [DOWN, RIGHT])
def test_moving_is_valid(action):
    *_, info = _env().step(action)
    assert not info["invalid_action"]


def test_pickup_and_dropoff_validity():
    env = _env()
    assert env.step(PICKUP)[-1]["invalid_action"]          # nothing here
    assert env.step(DROPOFF)[-1]["invalid_action"]         # empty-handed
    assert not pick_up(env, 0)[-1]["invalid_action"]
    assert env.step(PICKUP)[-1]["invalid_action"]          # hands full
    assert env.step(DROPOFF)[-1]["invalid_action"]         # not at the zone
    assert not deliver(env)[-1]["invalid_action"]


def test_progress_and_termination_reason():
    env = _env()
    pick_up(env, 0)
    *_, info = deliver(env)
    assert info["delivered"] == 1
    assert not info["is_success"] and info["termination_reason"] is None
    pick_up(env, 1)
    _, _, terminated, _, info = deliver(env)
    assert terminated
    assert info["delivered"] == 2
    assert info["is_success"] and info["termination_reason"] == "all_delivered"


def test_truncation_reason():
    env = _env(max_steps=3)
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(PICKUP)
    assert truncated and not terminated
    assert not info["is_success"] and info["termination_reason"] == "max_steps"


def test_success_on_last_step_reports_all_delivered():
    env = WarehouseEnv(grid_size=3, num_packages=1, max_steps=2)
    env.reset(seed=0)
    env.packages = [[2, 2, 0]]
    env.agent_pos = np.array([2, 2])
    env.step(PICKUP)
    _, _, terminated, truncated, info = env.step(DROPOFF)
    assert terminated and truncated
    assert info["termination_reason"] == "all_delivered"
