"""The lab trains any Gymnasium id; per-env profiles are optional."""

import json

import pytest

from lab_envs import EnvProfile, get_profile
from lab_render import ObservationRenderer
from train_agent import make_env, read_run_config, train_agent
from warehouse_lab import WAREHOUSE_PARAMS, WAREHOUSE_PROFILE


def test_unregistered_ids_get_the_default_profile():
    profile = get_profile("CartPole-v1")
    assert profile == EnvProfile("CartPole-v1")
    assert profile.params == ()
    assert isinstance(profile.renderer(), ObservationRenderer)
    assert profile.run_tag({}) == "CartPole-v1"
    assert get_profile("ale_py:ALE/Pong-v5").run_tag({}) == "ale-py-ALE-Pong-v5"


def test_warehouse_profile():
    from warehouse_render import WarehouseRenderer

    assert get_profile("Warehouse-v0") is WAREHOUSE_PROFILE
    assert WAREHOUSE_PROFILE.params == WAREHOUSE_PARAMS
    assert isinstance(WAREHOUSE_PROFILE.renderer(), WarehouseRenderer)
    assert WAREHOUSE_PROFILE.run_tag(dict(grid_size=8, num_packages=1)) == "g8_p1"


@pytest.mark.parametrize("env_id, kwargs, state_keys", [
    ("Warehouse-v0", dict(grid_size=5), {"grid_size", "agent_pos", "carrying", "packages",
                                         "delivery_zone"}),
    ("CartPole-v1", {}, {"observation", "low", "high"}),
])
def test_make_env_builds_any_id_with_a_render_state(env_id, kwargs, state_keys):
    venv = make_env(env_id, seed=0, **kwargs)
    venv.reset()
    _, _, _, infos = venv.step([0])
    assert set(infos[0]["render_state"]) == state_keys
    venv.close()


def test_run_config_records_the_env_id(tmp_path):
    _, _, env = train_agent(total_timesteps=50, seed=0, run_dir=tmp_path, checkpoint_freq=0)
    env.close()
    assert read_run_config(tmp_path)["env_id"] == "Warehouse-v0"
    assert json.loads((tmp_path / "config.json").read_text())["env"]["grid_size"] == 8


def test_dashboard_refuses_to_load_a_run_of_another_env(monkeypatch):
    pygame = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from ui_dashboard import Dashboard

    _, _, env = train_agent(total_timesteps=50, seed=0, checkpoint_freq=0)  # into ./runs/
    env.close()
    d = Dashboard(env_id="CartPole-v1")
    d._cmd_load()
    assert d._model_status == "Can't load: run is for Warehouse-v0"
    assert d._load_path is None
    pygame.quit()
