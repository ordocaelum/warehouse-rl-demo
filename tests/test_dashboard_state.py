"""Dashboard state transitions, driven headlessly (SDL dummy video driver)."""

import os
import time

import pytest

pygame = pytest.importorskip("pygame")

TIMESTEPS_SLIDER = 3  # value is in thousands
SEED_SLIDER = 9


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from ui_dashboard import Dashboard

    d = Dashboard()
    yield d
    d._cmd_stop()
    pygame.quit()


def _set_timesteps(d, thousands):
    d._sliders[TIMESTEPS_SLIDER]._value = thousands  # bypasses the slider minimum


def _wait_for(predicate, timeout=60):
    end = time.time() + timeout
    while not predicate():
        assert time.time() < end, "timed out"
        time.sleep(0.02)


def test_stop_ends_in_idle_with_model_available(dashboard):
    _set_timesteps(dashboard, 500)
    dashboard._cmd_start()
    _wait_for(lambda: dashboard._metrics_slot.puts > 0)
    dashboard._cmd_stop()
    assert dashboard._state == "idle"
    assert dashboard._model is not None
    assert dashboard._model_status == "Stopped (unsaved)"


def test_result_arriving_after_stop_returns_does_not_change_state(dashboard):
    _set_timesteps(dashboard, 500)
    dashboard._cmd_start()
    _wait_for(lambda: dashboard._metrics_slot.puts > 0)
    thread = dashboard._thread
    thread.join = lambda timeout=None: None  # as if join() had timed out
    dashboard._cmd_stop()
    assert dashboard._state == "idle"

    _wait_for(lambda: not thread.is_alive())  # the result arrives late
    dashboard._apply_finished()
    assert dashboard._state == "idle"
    assert dashboard._model_status == "Stopped (unsaved)"


def test_completed_run_ends_in_done(dashboard):
    _set_timesteps(dashboard, 1)
    dashboard._cmd_start()
    _wait_for(lambda: not dashboard._thread.is_alive())
    dashboard._apply_finished()
    assert dashboard._state == "done"
    assert dashboard._model is not None


def test_result_of_an_earlier_run_is_ignored(dashboard):
    _set_timesteps(dashboard, 1)
    dashboard._cmd_start()
    first = dashboard._thread
    _wait_for(lambda: not first.is_alive())
    dashboard._apply_finished()

    _set_timesteps(dashboard, 500)
    dashboard._cmd_start()
    dashboard._done_q.put(first)  # stale message
    dashboard._apply_finished()
    assert dashboard._state == "training"


def test_reward_history_gets_each_episode_and_reset_sticks(dashboard):
    _set_timesteps(dashboard, 500)
    dashboard._cmd_start()
    _wait_for(lambda: dashboard._episode_q.qsize() >= 3)
    dashboard._drain_episodes()
    before_reset = len(dashboard._ep_rewards)
    assert before_reset >= 3

    dashboard._cmd_reset()
    assert dashboard._ep_rewards == []
    _wait_for(lambda: dashboard._episode_q.qsize() >= 1)
    dashboard._cmd_pause()  # training blocks at its next step
    time.sleep(0.3)
    dashboard._drain_episodes()
    _, latest = dashboard._metrics_slot.take()

    # Only episodes finished after the reset are shown; the earlier ones do
    # not come back. (+1: on an episode's last step its return is queued
    # after that step's metrics were sent.)
    assert len(dashboard._ep_rewards) >= 1
    assert len(dashboard._ep_rewards) <= latest["episode"] + 1 - before_reset
