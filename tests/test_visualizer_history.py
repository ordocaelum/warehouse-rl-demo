"""The visualizer's reward history holds one value per finished episode."""

from fakes import RecordingVisualizer
from visualizer_pygame import WarehouseVisualizer


def test_per_step_updates_do_not_enter_history():
    viz = WarehouseVisualizer()  # not started: no window is opened
    for step, running_sum in enumerate([-0.1, -0.2, -0.3], start=1):
        viz._merge_state({"step": step, "ep_reward": running_sum})
    viz.put_episode_reward(9.5)
    viz.put_episode_reward(-3.0)
    assert viz.reward_history() == [9.5, -3.0]


def test_callback_reports_each_episode_return_once():
    from train_agent import train_agent

    viz = RecordingVisualizer()
    _, callback, env = train_agent(
        total_timesteps=300, visualizer=viz, seed=0,
        grid_size=5, num_packages=1, max_steps=30,
    )
    env.close()
    assert len(callback.episode_rewards) >= 5
    assert viz.episode_rewards == callback.episode_rewards
