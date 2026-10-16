"""Constructor arguments are validated instead of failing later."""

import numpy as np
import pytest

from warehouse_env import WarehouseEnv


@pytest.mark.parametrize("kwargs", [
    dict(grid_size=2), dict(grid_size=0), dict(grid_size=-5),
    dict(num_packages=0), dict(num_packages=-1),
    dict(max_steps=0),
    dict(step_cost=float("nan")), dict(step_cost=float("inf")),
])
def test_out_of_range_values_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        WarehouseEnv(**kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(grid_size=8.0), dict(grid_size="8"), dict(num_packages=True), dict(max_steps=1.5),
])
def test_wrong_types_raise_type_error(kwargs):
    with pytest.raises(TypeError):
        WarehouseEnv(**kwargs)


def test_smallest_valid_configuration_works():
    env = WarehouseEnv(grid_size=3, num_packages=1, max_steps=1, step_cost=0.0)
    env.reset(seed=0)
    env.step(0)


def test_numpy_integers_are_accepted():
    env = WarehouseEnv(grid_size=np.int64(5), num_packages=np.int32(2))
    assert type(env.grid_size) is int and env.grid_size == 5
