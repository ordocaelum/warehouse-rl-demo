"""The visualizer's S key (while paused) advances one episode, not one step."""

from fakes import SteppingVisualizer


def test_step_request_runs_to_end_of_episode():
    from train_agent import train_agent

    # max_steps=20 on 8x8: random play cannot deliver that fast with this seed,
    # so every episode is exactly 20 steps long.
    viz = SteppingVisualizer(presses=3)
    _, callback, env = train_agent(
        total_timesteps=200, visualizer=viz, seed=0,
        grid_size=8, num_packages=1, max_steps=20,
    )
    env.close()

    # Paused on the first step; each S then shows an episode's final frame.
    assert viz.blocked_at == [(0, 1), (0, 20), (1, 20)]
