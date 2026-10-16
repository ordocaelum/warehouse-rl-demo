"""Behaviour of the dashboard's background training thread."""

import queue
import threading

from helpers import thread_params
from ui_dashboard import _TrainingThread

PARAMS = dict(
    grid_size=4, num_packages=1, max_steps=50, total_timesteps=1000,
    learning_rate=1e-3, exploration_initial=1.0, exploration_final=0.1,
    buffer_size=1000, step_cost=0.05, seed=0, load_path=None,
)


def _run(stop_immediately, run_dir):
    stop_ev = threading.Event()
    if stop_immediately:
        stop_ev.set()
    results = []
    thread = _TrainingThread(
        thread_params(run_dir, **PARAMS), queue.Queue(), stop_ev, threading.Event(),
        on_done=lambda t: results.append((t.model, t.error)),
    )
    thread.start()
    thread.join(timeout=60)
    assert not thread.is_alive()
    (model, error), = results
    assert error is None
    return model


def test_stopped_run_saves_nothing(tmp_path):
    model = _run(stop_immediately=True, run_dir=tmp_path / "run")
    assert model is not None  # the partial model is still offered for SAVE
    assert not (tmp_path / "run" / "model.zip").exists()


def test_completed_run_saves_model_and_logs_in_its_run_folder(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _run(stop_immediately=False, run_dir=run)
    names = {p.name for p in run.iterdir()}
    assert {"model.zip", "config.json", "0.monitor.csv", "progress.csv"} <= names
    assert any(n.startswith("events.out.tfevents") for n in names)  # TensorBoard


def test_dashboard_and_cli_build_identical_agents(tmp_path):
    """The same visible settings must give the same DQN from either entry point."""
    from train_agent import train_agent

    ui_model = _run(stop_immediately=True, run_dir=tmp_path / "ui")
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
