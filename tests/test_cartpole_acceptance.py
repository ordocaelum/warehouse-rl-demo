"""Acceptance test: the training lab runs Gymnasium's stock CartPole-v1.

Nothing here is CartPole-specific except its id: no profile is registered,
no renderer or settings are passed, and the dashboard's own event loop runs
(headless) from START until training completes.
"""

import json
import threading
import time
from pathlib import Path

import pytest

pygame = pytest.importorskip("pygame")
np = pytest.importorskip("numpy")


def test_training_lab_runs_stock_cartpole(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from lab_params import AGENT_PARAMS, RUN_PARAMS
    from lab_render import ObservationRenderer
    from ui_dashboard import CENTER_W, LEFT_W, Dashboard

    d = Dashboard(env_id="CartPole-v1")
    # CartPole has no settings of its own: only the agent's and the run's.
    assert list(d._sliders) == [p.name for p in AGENT_PARAMS + RUN_PARAMS]
    assert isinstance(d._renderer, ObservationRenderer)
    d._sliders["total_timesteps"]._value = 2  # 2,000 steps (below the slider minimum)

    # Watch what the centre panel draws, without changing how it draws.
    drawn, bar_pixels = [], []
    real_draw = d._renderer.draw

    def watching_draw(surface, rect, state):
        result = real_draw(surface, rect, state)
        if state:
            drawn.append((rect, state, result))
            if not bar_pixels:
                px = pygame.surfarray.array3d(surface)[result.left:result.right,
                                                       result.top:result.bottom]
                bar_pixels.append(int(sum((px == c).all(-1).sum()
                                          for c in (ObservationRenderer.POS,
                                                    ObservationRenderer.NEG))))
        return result

    d._renderer.draw = watching_draw

    def quit_when_finished(deadline=time.time() + 120):
        while d._state in ("training", "paused") and time.time() < deadline:
            time.sleep(0.05)  # the UI loop applies the finished run
        pygame.event.post(pygame.event.Event(pygame.QUIT))

    d._cmd_start()
    threading.Thread(target=quit_when_finished, daemon=True).start()
    d.run()  # the real UI loop, on this (main) thread, until QUIT

    # Trained to completion and saved like any run.
    assert d._state == "done", d._err_msg
    assert d._model.num_timesteps == 2000
    run_dir = Path(d._run_dir)
    assert run_dir.name.endswith("_CartPole-v1_s0")
    assert (run_dir / "model.zip").exists()
    config = json.loads((run_dir / "config.json").read_text())
    assert config["env_id"] == "CartPole-v1" and config["env"] == {}

    # Displayed: CartPole's 4-component observation, drawn as bars in the centre panel.
    assert drawn
    for rect, state, result in drawn:
        assert len(state["observation"]) == 4
        assert LEFT_W <= rect.left and rect.right <= LEFT_W + CENTER_W
        assert result is not None
    assert bar_pixels[0] > 0

    # Episodes and the algorithm's own metrics reached the panels.
    assert len(d._ep_rewards) >= 10
    assert d._metrics["algo_metrics"]["train/n_updates"] > 0
    assert "rollout/exploration_rate" in d._metrics["algo_metrics"]
