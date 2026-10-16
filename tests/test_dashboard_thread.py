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


def test_dashboard_and_cli_build_identical_agents(tmp_path, monkeypatch):
    """The same visible settings must give the same DQN from either entry point."""
    from train_agent import train_agent

    monkeypatch.chdir(tmp_path)
    ui_model = _run(stop_immediately=True)
    cli_model, _, env = train_agent(
        total_timesteps=1,
        grid_size=PARAMS["grid_size"], num_packages=PARAMS["num_packages"],
        max_steps=PARAMS["max_steps"], step_cost=PARAMS["step_cost"],
        learning_rate=PARAMS["learning_rate"], buffer_size=PARAMS["buffer_size"],
        exploration_initial=PARAMS["exploration_initial"],
        exploration_final=PARAMS["exploration_final"], seed=PARAMS["seed"],
    )
    env.close()

    def settings(model):
        return {
            name: getattr(model, name)
            for name in (
                "learning_rate", "buffer_size", "learning_starts", "batch_size",
                "tau", "gamma", "gradient_steps", "target_update_interval",
                "exploration_fraction", "exploration_initial_eps",
                "exploration_final_eps", "max_grad_norm", "seed",
            )
        } | {"train_freq": tuple(model.train_freq), "net_arch": model.policy.net_arch}

    assert settings(ui_model) == settings(cli_model)
