"""Behaviour of the dashboard's background training thread."""

import os
import queue
import threading

from ui_dashboard import _TrainingThread

PARAMS = dict(
    grid_size=4, num_packages=1, max_steps=50, total_timesteps=1000,
    learning_rate=1e-3, exploration_initial=1.0, exploration_final=0.1,
    buffer_size=1000, step_cost=0.05, seed=0, load_path=None,
)


def _run(stop_immediately):
    stop_ev = threading.Event()
    if stop_immediately:
        stop_ev.set()
    results = []
    thread = _TrainingThread(
        dict(PARAMS), queue.Queue(), stop_ev, threading.Event(),
        on_done=lambda model, error: results.append((model, error)),
    )
    thread.start()
    thread.join(timeout=60)
    assert not thread.is_alive()
    (model, error), = results
    assert error is None
    return model


def test_stop_does_not_overwrite_saved_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "warehouse_dqn_model.zip").write_bytes(b"previous model")

    model = _run(stop_immediately=True)

    assert model is not None  # the partial model is still offered for SAVE
    assert (tmp_path / "warehouse_dqn_model.zip").read_bytes() == b"previous model"


def test_completed_run_saves_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(stop_immediately=False)
    assert os.path.exists(tmp_path / "warehouse_dqn_model.zip")
