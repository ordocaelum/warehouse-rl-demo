"""The declarative parameter schema and the dashboard sliders built from it."""

import pytest

from lab_params import AGENT_PARAMS, RUN_PARAMS, Param, by_target
from warehouse_lab import WAREHOUSE_PARAMS


@pytest.mark.parametrize("kwargs", [
    dict(target="world"),          # unknown target
    dict(type=str),                # not int/float
    dict(min=5, max=5),            # empty range
    dict(default=11),              # default outside range
    dict(log=True, min=0),         # log scale needs a positive range
])
def test_invalid_params_are_rejected(kwargs):
    spec = dict(name="x", target="env", type=int, min=0, max=10, default=1) | kwargs
    with pytest.raises(ValueError):
        Param(**spec)


def test_values_are_scaled_and_typed():
    p = Param("buffer_size", "agent", int, 1, 50, 10, scale=1000)
    assert p.value(12.0) == 12000 and type(p.value(12.0)) is int
    assert p.shown(12000) == 12
    assert p.in_range(50000) and not p.in_range(51000)
    assert p.label == "buffer_size"  # defaults to the name
    f = Param("lr", "agent", float, 1e-4, 1e-2, 1e-3, log=True)
    assert f.value(2e-3) == 2e-3 and type(f.value(1)) is float


def test_by_target_groups_values_by_name():
    params = WAREHOUSE_PARAMS + AGENT_PARAMS + RUN_PARAMS
    values = {p.name: p.value(p.default) for p in params}
    grouped = by_target(params, values)
    assert grouped["env"] == dict(grid_size=8, num_packages=1, max_steps=300, step_cost=0.05)
    assert grouped["agent"] == dict(learning_rate=1e-3, exploration_initial=1.0,
                                    exploration_final=0.1, buffer_size=10_000)
    assert grouped["run"] == dict(total_timesteps=100_000, seed=0)
    with pytest.raises(KeyError):
        by_target(params, dict(values, colour=1))
    with pytest.raises(KeyError):
        by_target(params, {k: v for k, v in values.items() if k != "seed"})


@pytest.fixture
def make_dashboard(monkeypatch):
    pygame = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from ui_dashboard import Dashboard

    yield Dashboard
    pygame.quit()


def test_dashboard_builds_one_slider_per_param_and_reads_them_by_name(make_dashboard):
    d = make_dashboard()
    assert [s.label for s in d._sliders.values()] == [
        p.label for p in WAREHOUSE_PARAMS + AGENT_PARAMS + RUN_PARAMS]
    d._sliders["grid_size"].set_value(11)
    d._sliders["buffer_size"].set_value(20)
    params = d._get_params()
    assert params["env"]["grid_size"] == 11
    assert params["agent"]["buffer_size"] == 20_000
    assert params["run"] == dict(total_timesteps=100_000, seed=0)


def test_dashboard_takes_any_env_params(make_dashboard):
    d = make_dashboard(env_params=(Param("length", "env", float, 0.5, 2.0, 1.0),))
    assert list(d._sliders)[0] == "length"
    assert d._get_params()["env"] == {"length": 1.0}
