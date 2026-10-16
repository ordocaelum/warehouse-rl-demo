"""test_agent evaluates on the configuration the model was trained on."""

import pytest

import train_agent as ta  # not "from ... import test_agent": pytest would collect it

CONFIG = dict(grid_size=5, num_packages=2, max_steps=30)


@pytest.fixture
def saved_model(tmp_path):
    _, _, env = ta.train_agent(total_timesteps=200, seed=0, run_dir=tmp_path, **CONFIG)
    env.close()
    return tmp_path / "model"


def test_evaluates_on_the_given_configuration(saved_model):
    rewards = ta.test_agent(saved_model, num_episodes=2, render=False, seed=0, **CONFIG)
    assert len(rewards) == 2


def test_mismatched_configuration_is_refused(saved_model):
    with pytest.raises(ValueError, match="different configuration"):
        ta.test_agent(saved_model, num_episodes=1, render=False)  # default 8x8, 1 pkg


def test_evaluation_is_reproducible(saved_model):
    r1 = ta.test_agent(saved_model, num_episodes=2, render=False, seed=4, **CONFIG)
    r2 = ta.test_agent(saved_model, num_episodes=2, render=False, seed=4, **CONFIG)
    assert r1 == r2
