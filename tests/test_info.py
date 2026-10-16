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


def test_delivery_zone_is_reported():
    env = WarehouseEnv(grid_size=7)
    _, info = env.reset(seed=0)
    assert info["delivery_zone"] == [6, 6]
    *_, info = env.step(RIGHT)
    assert info["delivery_zone"] == [6, 6]


def test_callbacks_forward_delivery_zone_from_info(monkeypatch):
    import queue
    import threading

    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env

    import train_agent
    from fakes import RecordingVisualizer
    from ui_dashboard import _Callback

    class MovedZone(WarehouseEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.delivery_zone = (0, self.grid_size - 1)

    viz = RecordingVisualizer()
    monkeypatch.setattr(train_agent, "WarehouseEnv", MovedZone)
    _, _, env = train_agent.train_agent(total_timesteps=50, visualizer=viz, seed=0, grid_size=5)
    env.close()
    assert {tuple(u["delivery_zone"]) for u in viz.updates} == {(0, 4)}

    q = queue.Queue()
    env = make_vec_env(lambda: MovedZone(grid_size=5), n_envs=1, seed=0)
    DQN("MlpPolicy", env, seed=0).learn(50, callback=_Callback(q, threading.Event(), threading.Event()))
    env.close()
    assert {tuple(u["delivery_zone"]) for u in q.queue} == {(0, 4)}
