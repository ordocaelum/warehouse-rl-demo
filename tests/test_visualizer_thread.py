"""The visualizer's window runs on the calling (main) thread."""

import threading

import pytest

pygame = pytest.importorskip("pygame")


def test_run_on_main_thread_while_training_thread_feeds_it(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from visualizer_pygame import WarehouseVisualizer
    from warehouse_render import WarehouseRenderer

    render_threads = []

    class RecordingRenderer(WarehouseRenderer):
        def draw(self, surface, rect, state):
            render_threads.append(threading.current_thread())
            return super().draw(surface, rect, state)

    viz = WarehouseVisualizer(grid_size=5, renderer=RecordingRenderer())

    def producer():
        for step in range(2000):
            viz.put_update({"step": step, "render_state": {
                "agent_pos": [step % 5, 0], "packages": [[2, 2, 0]],
                "carrying": -1, "grid_size": 5, "delivery_zone": [4, 4]}})
            if step % 500 == 0:
                viz.put_episode_reward(float(step))
        while not render_threads:  # make sure at least one frame was drawn
            pass
        viz.stop()

    trainer = threading.Thread(target=producer)
    trainer.start()
    viz.run()  # returns once the producer calls stop()
    trainer.join(timeout=10)

    assert not viz.is_running()
    assert render_threads and all(t is threading.main_thread() for t in render_threads)
    assert viz.reward_history() == [0.0, 500.0, 1000.0, 1500.0]
