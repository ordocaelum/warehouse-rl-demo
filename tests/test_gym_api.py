"""Gymnasium API conventions: registration, render modes, action masks."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import warehouse_env  # noqa: F401  (registers Warehouse-v0)
from helpers import DROPOFF, PICKUP, RIGHT, deliver, pick_up
from warehouse_env import WarehouseEnv


def test_gym_make():
    env = gym.make("Warehouse-v0", grid_size=5, num_packages=2)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (9,)
    assert isinstance(env.unwrapped, WarehouseEnv)
    env.close()


def test_check_env_with_render_modes():
    # Through gym.make, so the checker also exercises every render mode.
    check_env(gym.make("Warehouse-v0", render_mode="ansi").unwrapped)


def test_ansi_render_returns_text():
    env = WarehouseEnv(grid_size=4, render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    rows = text.splitlines()
    assert rows[0][0] == "A" and rows[3][3] == "D"
    assert "Step: 0" in text


def test_human_render_prints_on_reset_and_step(capsys):
    env = WarehouseEnv(grid_size=4, render_mode="human")
    env.reset(seed=0)
    env.step(RIGHT)
    assert capsys.readouterr().out.count("Step:") == 2


def test_no_render_mode_renders_nothing(capsys):
    env = WarehouseEnv()
    env.reset(seed=0)
    with pytest.warns(UserWarning, match="render_mode"):
        assert env.render() is None
    assert capsys.readouterr().out == ""


def test_unknown_render_mode_is_rejected():
    with pytest.raises(ValueError):
        WarehouseEnv(render_mode="rgb_array")


def test_action_masks_match_invalid_action():
    env = WarehouseEnv(grid_size=5, num_packages=2)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        mask = env.action_masks()
        action = int(rng.integers(6))
        _, _, terminated, truncated, info = env.step(action)
        assert info["invalid_action"] == (not mask[action])
        np.testing.assert_array_equal(info["action_mask"], env.action_masks())
        if terminated or truncated:
            env.reset()


def test_action_masks_for_pickup_and_dropoff():
    env = WarehouseEnv(grid_size=6, num_packages=1)
    env.reset(seed=0)
    assert not env.action_masks()[PICKUP]
    pick_up(env, 0)
    assert not env.action_masks()[PICKUP]
    assert not env.action_masks()[DROPOFF] or tuple(env.agent_pos) == env.delivery_zone
    *_, info = deliver(env)
    assert not info["action_mask"][DROPOFF]
