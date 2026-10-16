"""The last step of each episode (e.g. the delivery) must reach the display."""

import queue
import threading

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from fakes import RecordingVisualizer
from warehouse_env import WarehouseEnv

# A 3x3 grid has a single package cell, so random actions deliver often.
# The tests act fully at random (epsilon 1.0) so that deliveries happen.
EASY = dict(grid_size=3, num_packages=1, max_steps=60)


def _all_delivered(update):
    return all(p[2] for p in update["packages"])


def test_state_from_observation_round_trips():
    env = WarehouseEnv(grid_size=6, num_packages=2)
    env.reset(seed=5)
    rng = np.random.default_rng(5)
    for _ in range(300):
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(6)))
        state = WarehouseEnv.state_from_observation(obs)
        assert state["agent_pos"] == env.agent_pos.tolist()
        assert state["packages"] == env.packages
        assert (state["carrying"] >= 0) == (env.carrying >= 0)
        if terminated or truncated:
            env.reset()


def test_cli_callback_shows_delivery():
    from train_agent import train_agent

    viz = RecordingVisualizer()
    model, callback, env = train_agent(
        total_timesteps=1000, visualizer=viz, seed=0,
        exploration_initial=1.0, exploration_final=1.0, **EASY,
    )
    env.close()
    assert callback.episode_rewards, "no episode finished"
    assert any(_all_delivered(u) for u in viz.updates)


def test_dashboard_callback_shows_delivery():
    from ui_dashboard import _Callback

    q = queue.Queue()
    env = make_vec_env(lambda: WarehouseEnv(**EASY), n_envs=1, seed=0)
    model = DQN("MlpPolicy", env, learning_starts=100, seed=0,
                exploration_initial_eps=1.0, exploration_final_eps=1.0)
    model.learn(1000, callback=_Callback(q, threading.Event(), threading.Event()))
    env.close()
    updates = list(q.queue)
    assert any(_all_delivered(u) for u in updates)
