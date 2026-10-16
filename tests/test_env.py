"""Environment contract: observation, rewards, termination and truncation."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from helpers import DOWN, DROPOFF, LEFT, PICKUP, RIGHT, UP, deliver, pick_up, walk_to
from warehouse_env import WarehouseEnv

STEP_COST = 0.05


def _env_with(packages, agent=(0, 0), **kwargs):
    """An env in a hand-made state: packages [[x, y, delivered], ...]."""
    env = WarehouseEnv(step_cost=STEP_COST, num_packages=len(packages), **kwargs)
    env.reset(seed=0)
    env.packages = [list(p) for p in packages]
    env.agent_pos = np.array(agent)
    env.last_distance = env._get_nearest_package_distance()
    return env


# -- observation ------------------------------------------------------------

@pytest.mark.parametrize("num_packages, dims", [(1, 6), (2, 9), (5, 18)])
def test_observation_shape_and_dtype(num_packages, dims):
    env = WarehouseEnv(num_packages=num_packages)
    obs, info = env.reset(seed=0)
    assert env.observation_space.shape == (dims,)
    assert obs.shape == (dims,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_observations_stay_inside_observation_space():
    env = WarehouseEnv(grid_size=6, num_packages=3, max_steps=50)
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(500):
        assert env.observation_space.contains(obs)
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(6)))
        if terminated or truncated:
            obs, _ = env.reset()


def test_observation_layout():
    env = _env_with([[2, 3, 0], [4, 1, 1]], agent=(5, 6))
    np.testing.assert_array_equal(env._get_obs(), [5, 6, 0, 2, 3, 0, 4, 1, 1])


def test_gymnasium_env_checker_passes():
    check_env(WarehouseEnv(), skip_render_check=True)


# -- rewards, term by term ----------------------------------------------------

def test_step_cost_alone_for_a_no_op():
    env = _env_with([[3, 3, 0]])
    _, reward, *_ = env.step(PICKUP)  # nothing here to pick up
    assert reward == pytest.approx(-STEP_COST)


def test_pickup_reward():
    env = _env_with([[3, 3, 0]], agent=(3, 3))
    _, reward, *_ = env.step(PICKUP)
    assert reward == pytest.approx(-STEP_COST + 1)
    assert env.carrying == 0


def test_delivery_reward():
    env = _env_with([[3, 3, 0]], agent=(3, 3))
    env.step(PICKUP)
    _, reward, *_ = deliver(env)
    assert reward == pytest.approx(-STEP_COST + 10)
    assert env.packages[0][2] == 1


def test_approach_bonus_and_distance_penalty():
    env = _env_with([[3, 3, 0]], agent=(0, 0))  # distance 6
    _, reward, *_ = env.step(RIGHT)  # distance 5: closer
    assert reward == pytest.approx(-STEP_COST + 0.1 - 0.01 * 5)
    _, reward, *_ = env.step(LEFT)  # distance 6: farther, no bonus
    assert reward == pytest.approx(-STEP_COST - 0.01 * 6)


def test_wall_bump_gets_distance_penalty_but_no_bonus():
    env = _env_with([[3, 3, 0]], agent=(0, 0))
    _, reward, *_ = env.step(UP)  # already at the top edge
    assert env.agent_pos.tolist() == [0, 0]
    assert reward == pytest.approx(-STEP_COST - 0.01 * 6)


def test_no_distance_shaping_while_carrying():
    env = _env_with([[3, 3, 0]], agent=(3, 3))
    env.step(PICKUP)
    for action in (DOWN, UP, RIGHT):
        _, reward, *_ = env.step(action)
        assert reward == pytest.approx(-STEP_COST)


def test_dropoff_away_from_delivery_zone_does_nothing():
    env = _env_with([[3, 3, 0]], agent=(3, 3))
    env.step(PICKUP)
    _, reward, *_ = env.step(DROPOFF)
    assert reward == pytest.approx(-STEP_COST)
    assert env.carrying == 0 and env.packages[0][2] == 0


def test_delivered_package_cannot_be_picked_up_again():
    env = _env_with([[3, 3, 0], [1, 1, 0]], agent=(3, 3))
    env.step(PICKUP)
    deliver(env)
    _, reward, *_ = env.step(PICKUP)  # standing on the delivered package
    assert reward == pytest.approx(-STEP_COST)
    assert env.carrying == -1


# -- termination and truncation ---------------------------------------------

def test_terminates_only_when_every_package_is_delivered():
    env = _env_with([[3, 3, 0], [2, 5, 0]], agent=(3, 3))
    env.step(PICKUP)
    _, _, terminated, truncated, _ = deliver(env)
    assert not terminated and not truncated
    pick_up(env, 1)
    _, _, terminated, truncated, _ = deliver(env)
    assert terminated and not truncated


def test_truncates_at_max_steps():
    env = _env_with([[3, 3, 0]], max_steps=5)
    for i in range(1, 6):
        _, _, terminated, truncated, _ = env.step(PICKUP)
        assert not terminated
        assert truncated == (i == 5)


def test_delivery_on_the_last_step_is_both_terminated_and_truncated():
    env = _env_with([[7, 7, 0]], agent=(7, 7), max_steps=2)
    env.step(PICKUP)
    _, _, terminated, truncated, _ = env.step(DROPOFF)
    assert terminated and truncated


def test_reset_restores_start_state():
    env = WarehouseEnv(num_packages=2)
    env.reset(seed=3)
    pick_up(env, 0)
    env.reset(seed=3)
    assert env.agent_pos.tolist() == [0, 0]
    assert env.carrying == -1
    assert env.step_count == 0
    assert all(p[2] == 0 for p in env.packages)


# -- bounds and dtypes --------------------------------------------------------

def test_observation_bounds_are_exact():
    env = WarehouseEnv(grid_size=6, num_packages=2)
    np.testing.assert_array_equal(env.observation_space.low, np.zeros(9))
    np.testing.assert_array_equal(env.observation_space.high, [5, 5, 1, 5, 5, 1, 5, 5, 1])


def test_observation_reaches_its_upper_bound():
    # Carrying a package into the far corner puts every coordinate at
    # grid_size - 1 and the carrying flag at 1; delivering sets the last flag.
    env = _env_with([[3, 3, 0]], agent=(3, 3), grid_size=6)
    env.step(PICKUP)
    obs, *_ = walk_to(env, 5, 5)[-1]
    np.testing.assert_array_equal(obs[:5], env.observation_space.high[:5])
    obs, *_ = env.step(DROPOFF)
    assert obs[5] == env.observation_space.high[5]


def test_internal_types():
    env = WarehouseEnv(num_packages=2)
    env.reset(seed=0)
    _, reward, *_ = env.step(RIGHT)
    assert env.agent_pos.dtype == np.int64
    assert all(type(v) is int for p in env.packages for v in p)
    assert type(reward) is float
