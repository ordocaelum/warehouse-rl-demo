"""Algorithm metrics are forwarded by name; none (epsilon included) is assumed."""

import queue
import threading

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from fakes import RecordingVisualizer
from lab_metrics import collect_algorithm_metrics, display_metrics
from warehouse_env import WarehouseEnv


class _Logger:
    def __init__(self, values):
        self.name_to_value = values


def test_collect_keeps_latest_numeric_values_between_dumps():
    kept = {}
    collect_algorithm_metrics(_Logger({"a/x": 1, "b/y": np.float32(0.5), "c/s": "text"}), kept)
    collect_algorithm_metrics(_Logger({"a/x": 2.0}), kept)  # after a dump cleared b/y
    assert kept == {"a/x": 2.0, "b/y": 0.5}
    assert all(type(v) is float for v in kept.values())


def test_display_uses_short_names_in_recording_order():
    metrics = {"rollout/exploration_rate": 0.81234, "train/loss": 0.00123, "train/n_updates": 40.0}
    assert display_metrics(metrics, 2) == [("exploration_rate", "0.8123"), ("loss", "0.00123")]


def test_dqn_reports_exploration_rate_by_name():
    from train_agent import train_agent

    viz = RecordingVisualizer()
    _, _, env = train_agent(total_timesteps=700, visualizer=viz, seed=0, grid_size=5,
                            checkpoint_freq=0)
    env.close()
    assert all("epsilon" not in u for u in viz.updates)
    last = viz.updates[-1]["algo_metrics"]
    assert 0.0 <= last["rollout/exploration_rate"] <= 1.0
    assert "train/loss" in last  # training started after learning_starts=500


def test_dashboard_callback_with_an_algorithm_that_has_no_epsilon():
    from ui_dashboard import _Callback

    q = queue.Queue()
    env = make_vec_env(lambda: WarehouseEnv(grid_size=5), n_envs=1, seed=0)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0)
    assert not hasattr(model, "exploration_rate")
    model.learn(200, callback=_Callback(q, threading.Event(), threading.Event()))
    env.close()
    updates = list(q.queue)
    assert len(updates) == 256  # PPO collects whole 64-step rollouts
    assert all("epsilon" not in u for u in updates)
    last = updates[-1]["algo_metrics"]
    assert {"train/entropy_loss", "train/approx_kl"} <= set(last)


@pytest.mark.parametrize("algo_metrics", [None, {}, {"train/entropy_loss": -0.69, "train/approx_kl": 0.01}])
def test_panels_draw_with_any_or_no_algorithm_metrics(monkeypatch, algo_metrics):
    pygame = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from ui_dashboard import Dashboard
    from visualizer_pygame import WarehouseVisualizer

    d = Dashboard()
    d._metrics = {"episode": 3, "step": 7, "ep_reward": -1.0, "algo_metrics": algo_metrics}
    d._draw_right_panel()

    viz = WarehouseVisualizer()
    viz._merge_state({"algo_metrics": algo_metrics})
    font = pygame.font.SysFont("monospace", 12)
    viz._draw_panel(pygame.Surface((800, 480)), font, font, font, 480, 800, 480)
    pygame.quit()
