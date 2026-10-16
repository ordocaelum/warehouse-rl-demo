"""Checkpoints, saving during training, and resuming a saved model."""

import queue
import threading
import time

import train_agent as ta
from latest_slot import LatestSlot
from ui_dashboard import _TrainingThread

PARAMS = dict(
    grid_size=4, num_packages=1, max_steps=50, total_timesteps=200_000,
    learning_rate=1e-3, exploration_initial=1.0, exploration_final=0.1,
    buffer_size=5000, step_cost=0.05, seed=0, load_path=None,
)


def _wait_for(predicate, timeout=60):
    end = time.time() + timeout
    while not predicate():
        assert time.time() < end, "timed out"
        time.sleep(0.02)


def _start(run_dir, pause=False, **overrides):
    """Start a dashboard training thread; return (thread, stop, pause, save_req, saved_q)."""
    stop_ev, pause_ev, save_req, saved_q = (
        threading.Event(), threading.Event(), threading.Event(), queue.Queue())
    if pause:
        pause_ev.set()
    done = []
    thread = _TrainingThread(
        dict(PARAMS, run_dir=str(run_dir), **overrides), LatestSlot(), stop_ev, pause_ev,
        on_done=done.append, save_req=save_req, saved_q=saved_q,
    )
    thread.start()
    return thread, stop_ev, pause_ev, save_req, saved_q


def test_cli_writes_periodic_checkpoints_with_replay_buffer(tmp_path):
    _, _, env = ta.train_agent(total_timesteps=600, seed=0, run_dir=tmp_path,
                               grid_size=4, checkpoint_freq=200)
    env.close()
    names = {p.name for p in (tmp_path / "checkpoints").iterdir()}
    for steps in (200, 400, 600):
        assert f"model_{steps}_steps.zip" in names
        assert f"model_replay_buffer_{steps}_steps.pkl" in names
    assert (tmp_path / "model_replay_buffer.pkl").exists()


def test_save_during_training(tmp_path):
    thread, stop_ev, _, save_req, saved_q = _start(tmp_path)
    _wait_for(lambda: thread.model is not None and thread.model.num_timesteps > 100)
    save_req.set()
    saved_thread, step = saved_q.get(timeout=30)
    assert saved_thread is thread and step > 100
    assert (tmp_path / "model.zip").exists()
    assert (tmp_path / "model_replay_buffer.pkl").exists()
    assert thread.is_alive()  # still training
    stop_ev.set()
    thread.join(timeout=30)


def test_save_while_paused(tmp_path):
    thread, stop_ev, pause_ev, save_req, saved_q = _start(tmp_path, pause=True)
    _wait_for(lambda: thread.model is not None)
    time.sleep(0.3)  # blocked in the pause loop on its first step
    save_req.set()
    _, step = saved_q.get(timeout=30)
    assert step <= 1
    assert (tmp_path / "model.zip").exists()
    stop_ev.set()
    thread.join(timeout=30)


def test_load_resumes_the_saved_run(tmp_path):
    first = tmp_path / "first"
    _, _, env = ta.train_agent(total_timesteps=1000, seed=0, run_dir=first, grid_size=4,
                               max_steps=50, buffer_size=5000, learning_rate=5e-4,
                               checkpoint_freq=0)
    env.close()

    # Different agent settings on the sliders must not apply to a resumed run.
    thread, *_ = _start(tmp_path / "second", total_timesteps=500,
                        load_path=str(first / "model"), learning_rate=1e-2)
    _wait_for(lambda: thread.model is not None)
    resumed = thread.model
    thread.join(timeout=60)
    assert thread.error is None
    assert resumed.learning_rate == 5e-4
    assert resumed.num_timesteps == 1500          # step count carried on
    assert resumed.replay_buffer.size() == 1500   # buffer restored, then filled
    assert (tmp_path / "second" / "model.zip").exists()
