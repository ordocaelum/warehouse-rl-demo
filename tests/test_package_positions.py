"""Package coordinates must follow the package: carried, then delivered."""

from helpers import pick_up, deliver, walk_to, UP
from warehouse_env import WarehouseEnv


def _pkg_obs(obs, k):
    return list(obs[3 + 3 * k: 6 + 3 * k])


def test_carried_package_reports_agent_position():
    env = WarehouseEnv(grid_size=8, num_packages=2)
    env.reset(seed=0)
    pick_up(env, 0)
    obs, *_ = walk_to(env, 0, 0)[-1]
    assert env.packages[0][:2] == [0, 0]
    assert _pkg_obs(obs, 0) == [0, 0, 0]
    # The package that is not carried stays put.
    assert env.packages[1][:2] == list(map(int, _pkg_obs(obs, 1)[:2]))


def test_delivered_package_reports_delivery_zone():
    env = WarehouseEnv(grid_size=8, num_packages=2)
    env.reset(seed=0)
    pick_up(env, 0)
    obs, *_ = deliver(env)
    zx, zy = env.delivery_zone
    assert _pkg_obs(obs, 0) == [zx, zy, 1]
    # It stays there after the agent walks away.
    obs, *_ = env.step(UP)
    assert _pkg_obs(obs, 0) == [zx, zy, 1]
