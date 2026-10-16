"""Reproducibility: identical seeds must give identical environments and runs."""

import numpy as np
import pytest

from warehouse_env import WarehouseEnv


def _rollout(env, seed, n_steps=200):
    """Run a fixed-length rollout whose actions come from a seeded generator."""
    obs, _ = env.reset(seed=seed)
    actions = np.random.default_rng(seed).integers(0, env.action_space.n, n_steps)
    trajectory = [obs]
    for a in actions:
        obs, reward, terminated, truncated, _ = env.step(int(a))
        trajectory.append((obs, reward, terminated, truncated))
        if terminated or truncated:
            obs, _ = env.reset()
            trajectory.append(obs)
    return trajectory


def _assert_same(t1, t2):
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        if isinstance(a, tuple):
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:] == b[1:]
        else:
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("num_packages", [1, 3])
def test_same_seed_reset_gives_identical_observation(num_packages):
    env = WarehouseEnv(num_packages=num_packages)
    first, _ = env.reset(seed=42)
    for _ in range(5):
        env.step(env.action_space.sample())
    second, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)

    # A separately constructed env with the same seed agrees too.
    other, _ = WarehouseEnv(num_packages=num_packages).reset(seed=42)
    np.testing.assert_array_equal(first, other)


def test_different_seeds_give_different_layouts():
    env = WarehouseEnv(num_packages=3)
    layouts = {tuple(env.reset(seed=s)[0]) for s in range(20)}
    assert len(layouts) > 1


def test_same_seed_rollouts_are_identical():
    # Long enough to cross several episode boundaries, so unseeded resets
    # after the first seeded one are covered as well.
    kwargs = dict(grid_size=6, num_packages=2, max_steps=40)
    t1 = _rollout(WarehouseEnv(**kwargs), seed=7, n_steps=300)
    t2 = _rollout(WarehouseEnv(**kwargs), seed=7, n_steps=300)
    _assert_same(t1, t2)


def test_env_does_not_touch_global_numpy_rng():
    np.random.seed(123)
    expected = np.random.random(5)

    np.random.seed(123)
    env = WarehouseEnv(num_packages=3)
    env.reset(seed=1)
    env.reset()
    for _ in range(50):
        env.step(env.action_space.sample())
    np.testing.assert_array_equal(np.random.random(5), expected)


def test_seeded_training_runs_are_identical():
    """Two short DQN runs with the same seed end with identical weights.

    Covers env seeding, exploration and torch initialisation/updates on CPU.
    Determinism on GPU is not claimed (see README, Reproducibility).
    """
    torch = pytest.importorskip("torch")
    from train_agent import train_agent

    def run():
        model, callback, env = train_agent(total_timesteps=1500, seed=3)
        env.close()
        return model.policy.state_dict(), callback.episode_rewards

    params1, rewards1 = run()
    params2, rewards2 = run()

    assert rewards1 == rewards2
    assert params1.keys() == params2.keys()
    for name in params1:
        assert torch.equal(params1[name], params2[name]), name
