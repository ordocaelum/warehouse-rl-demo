"""The env pushes its render state; callbacks forward it and never read the env."""

import json
import queue
import threading

import numpy as np
import pytest
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from fakes import RecordingVisualizer
from helpers import DOWN, DROPOFF, PICKUP, RIGHT
from warehouse_env import WarehouseEnv

KEYS = {"grid_size", "agent_pos", "carrying", "packages", "delivery_zone"}


def _matches_env(state, env):
    return state == {
        "grid_size": env.grid_size,
        "agent_pos": env.agent_pos.tolist(),
        "carrying": env.carrying,
        "packages": env.packages,
        "delivery_zone": list(env.delivery_zone),
    }


def test_reset_and_step_report_the_new_state():
    env = WarehouseEnv(grid_size=6, num_packages=2)
    _, info = env.reset(seed=4)
    assert set(info["render_state"]) == KEYS
    assert _matches_env(info["render_state"], env)
    rng = np.random.default_rng(4)
    for _ in range(200):
        *_, terminated, truncated, info = env.step(int(rng.integers(6)))
        assert _matches_env(info["render_state"], env)
        if terminated or truncated:
            env.reset()


def test_render_state_is_plain_data():
    env = WarehouseEnv(grid_size=5, num_packages=3)
    _, info = env.reset(seed=0)
    state = info["render_state"]
    assert json.loads(json.dumps(state)) == state  # no numpy types, no objects


def test_render_state_is_a_snapshot_not_a_view():
    env = WarehouseEnv(grid_size=3, num_packages=1)
    _, info = env.reset(seed=0)
    before = info["render_state"]
    expected = json.loads(json.dumps(before))
    for action in (RIGHT, DOWN, PICKUP, RIGHT):  # the package moves with the agent
        env.step(action)
    assert before == expected  # later steps did not change the earlier snapshot

    *_, info = env.step(DOWN)
    info["render_state"]["packages"][0][0] = 99
    info["render_state"]["agent_pos"][0] = 99
    assert env.packages[0][0] == 2 and env.agent_pos[0] == 2  # nor does editing it


def test_done_step_reports_the_final_frame_after_auto_reset():
    # 3x3: the only package cell is (1, 1), the delivery zone (2, 2).
    venv = DummyVecEnv([lambda: WarehouseEnv(grid_size=3, num_packages=1)])
    venv.reset()
    for action in (RIGHT, DOWN, PICKUP, RIGHT, DOWN, DROPOFF):
        _, _, dones, infos = venv.step(np.array([action]))
    assert dones[0]
    assert infos[0]["render_state"]["agent_pos"] == [2, 2]
    assert infos[0]["render_state"]["packages"] == [[2, 2, 1]]
    # ...while the env itself has already started the next episode.
    assert venv.envs[0].unwrapped.agent_pos.tolist() == [0, 0]
    venv.close()


def test_callbacks_work_without_access_to_the_env_object():
    """In a subprocess env there is no Python env object for a callback to read."""
    from train_agent import RewardCallback
    from ui_dashboard import _Callback

    venv = make_vec_env(
        lambda: WarehouseEnv(grid_size=3, num_packages=1, max_steps=40),
        n_envs=1, seed=0, vec_env_cls=SubprocVecEnv,
    )
    assert not hasattr(venv, "envs")
    try:
        viz = RecordingVisualizer()
        DQN("MlpPolicy", venv, learning_starts=100, seed=0).learn(
            300, callback=RewardCallback(visualizer=viz))
        q = queue.Queue()
        DQN("MlpPolicy", venv, learning_starts=100, seed=0).learn(
            300, callback=_Callback(q, threading.Event(), threading.Event()))
    finally:
        venv.close()

    for updates in (viz.updates, list(q.queue)):
        assert len(updates) == 300
        assert all(set(u["render_state"]) == KEYS for u in updates)


def test_envs_without_render_state_report_their_observation():
    import gymnasium as gym

    from lab_envs import ObservationRenderState

    env = ObservationRenderState(gym.make("CartPole-v1"))
    obs, info = env.reset(seed=0)
    state = info["render_state"]
    assert state["observation"] == [float(v) for v in obs]
    assert state["low"][1] is None and state["high"][3] is None  # unbounded velocities
    assert state["low"][0] == pytest.approx(-4.8) and state["high"][0] == pytest.approx(4.8)
    assert json.loads(json.dumps(state)) == state
    obs, *_, info = env.step(0)
    assert info["render_state"]["observation"] == [float(v) for v in obs]
    env.close()


def test_own_render_state_passes_through_the_fallback():
    from lab_envs import ObservationRenderState

    _, info = ObservationRenderState(WarehouseEnv(grid_size=4)).reset(seed=0)
    assert set(info["render_state"]) == KEYS


def test_fallback_render_state_on_the_done_step_is_the_terminal_observation():
    import gymnasium as gym

    from lab_envs import ObservationRenderState

    venv = DummyVecEnv([lambda: ObservationRenderState(gym.make("CartPole-v1"))])
    venv.seed(0)
    venv.reset()
    dones = [False]
    while not dones[0]:
        _, _, dones, infos = venv.step(np.array([0]))  # always push left: falls fast
    assert infos[0]["render_state"]["observation"] == [
        float(v) for v in infos[0]["terminal_observation"]]
    venv.close()
