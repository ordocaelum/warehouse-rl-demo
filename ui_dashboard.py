"""
Warehouse RL Training Lab  --  Interactive PyGame Dashboard
===========================================================
Run:
    python ui_dashboard.py

Keyboard shortcuts:
    SPACE     Pause / resume training
    S         Save model
    R         Reset stats
    Q / ESC   Quit
"""

from __future__ import annotations

import math
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pygame
except ImportError as exc:
    raise SystemExit(
        "pygame is required.  Install it with:  pip install pygame"
    ) from exc

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
    from stable_baselines3.common.callbacks import CallbackList
    from train_agent import (
        EXPLORATION_FRACTION, LEARNING_STARTS, RUNS_DIR, TARGET_UPDATE_INTERVAL,
        attach_run_logger, checkpoint_callback, find_latest_model, load_run_model,
        make_dqn, make_env, new_run_dir, read_run_config, save_run_model,
        write_run_config,
    )
    _SB3_AVAILABLE = True
except ImportError:
    DQN = None          # type: ignore[assignment,misc]
    make_dqn = make_env = None  # type: ignore[assignment]
    _SB3_AVAILABLE = False

    class _BaseCallback:  # type: ignore[no-redef]
        """Stub so the _Callback class definition does not crash at import."""
        def __init__(self, verbose: int = 0) -> None:
            pass

from lab_metrics import collect_algorithm_metrics, display_metrics
from lab_params import AGENT_PARAMS, RUN_PARAMS, Param, by_target
from lab_render import Renderer
from latest_slot import LatestSlot
from lab_envs import get_profile
from warehouse_lab import WAREHOUSE_ID  # also registers Warehouse-v0 and its profile


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
SCREEN_W = 1300
SCREEN_H = 780
LEFT_W   = 275   # left parameter panel
RIGHT_W  = 305   # right stats / graph panel
BOTTOM_H = 115   # bottom button bar
CENTER_W = SCREEN_W - LEFT_W - RIGHT_W   # 720 px
CENTER_H = SCREEN_H - BOTTOM_H           # 665 px
FPS_CAP  = 60
GRAPH_EPISODES = 100  # episodes shown in the reward history graph
ALGO_ROWS = 4         # algorithm metrics shown in the right panel

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PAL: Dict[str, Tuple[int, int, int]] = {
    "bg":           ( 18,  20,  32),
    "panel":        ( 24,  27,  44),
    "border":       ( 55,  65,  98),
    "header":       (140, 165, 255),
    "text":         (190, 200, 225),
    "dim":          ( 88,  98, 118),
    "pos":          ( 75, 210,  95),
    "neg":          (220,  70,  70),
    "warn":         (230, 180,  50),
    "changed":      (255, 175,  40),
    "slider_bg":    ( 45,  50,  74),
    "slider_fill":  ( 80, 125, 215),
    "slider_knob":  (200, 215, 255),
    "btn":          ( 50,  58,  94),
    "btn_hover":    ( 70,  82, 132),
    "btn_start":    ( 40, 125,  68),
    "btn_stop":     (160,  42,  42),
    "btn_save":     ( 40, 115, 152),
    "graph_raw":    ( 90, 150, 232),
    "graph_avg":    (235, 162,  50),
    "graph_bg":     ( 18,  22,  38),
    "graph_grid":   ( 38,  44,  68),
    "paused_bg":    (200, 120,  30),
    "paused_text":  (255, 255, 255),
}


def _c(key: str) -> Tuple[int, int, int]:
    return _PAL[key]


# ---------------------------------------------------------------------------
# Slider widget
# ---------------------------------------------------------------------------
class Slider:
    """Horizontal draggable slider with label and value readout."""

    HEIGHT = 46  # bounding-box height per slider instance

    def __init__(
        self,
        x: int, y: int, w: int,
        label: str,
        min_val: float, max_val: float, default: float,
        is_int: bool = False,
        log_scale: bool = False,
        fmt: Optional[str] = None,
    ) -> None:
        self.rect      = pygame.Rect(x, y, w, self.HEIGHT)
        self._tx       = x + 6        # track left x
        self._tw       = w - 12       # track width
        self._ty       = y + 34       # track centre y
        self.label     = label
        self.min_val   = float(min_val)
        self.max_val   = float(max_val)
        self.is_int    = is_int
        self.log_scale = log_scale
        self.fmt       = fmt
        self._drag     = False
        self._changed  = False        # unsaved-change indicator
        self._value    = float(default)
        self._frac     = self._v2f(float(default))

    # -- value helpers --------------------------------------------------------
    def _v2f(self, v: float) -> float:
        v = max(self.min_val, min(self.max_val, v))
        if self.log_scale:
            lmn = math.log10(self.min_val)
            lmx = math.log10(self.max_val)
            return (math.log10(v) - lmn) / (lmx - lmn)
        return (v - self.min_val) / (self.max_val - self.min_val)

    def _f2v(self, f: float) -> float:
        f = max(0.0, min(1.0, f))
        if self.log_scale:
            lmn = math.log10(self.min_val)
            lmx = math.log10(self.max_val)
            v = 10 ** (lmn + f * (lmx - lmn))
        else:
            v = self.min_val + f * (self.max_val - self.min_val)
        return float(round(v) if self.is_int else v)

    @property
    def value(self) -> float:
        return self._value

    def _fmt_value(self) -> str:
        if self.fmt:
            return self.fmt.format(self._value)
        if self.is_int:
            return str(int(self._value))
        if self.log_scale:
            return f"{self._value:.2e}"
        return f"{self._value:.3f}"

    # -- events ---------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if the value changed."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            kx = self._tx + int(self._frac * self._tw)
            hit_knob = abs(event.pos[0] - kx) <= 12 and abs(event.pos[1] - self._ty) <= 12
            if hit_knob or self.rect.collidepoint(event.pos):
                self._drag = True
                return self._move(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag = False
        elif event.type == pygame.MOUSEMOTION and self._drag:
            return self._move(event.pos[0])
        return False

    def _move(self, mx: int) -> bool:
        nf = (mx - self._tx) / max(1, self._tw)
        nv = self._f2v(nf)
        if nv != self._value:
            self._value = nv
            self._frac  = max(0.0, min(1.0, nf))
            self._changed = True
            return True
        return False

    def mark_applied(self) -> None:
        """Clear the unsaved-change indicator once settings are applied."""
        self._changed = False

    def set_value(self, v: float) -> None:
        """Set the value programmatically (clamped to the slider's range)."""
        v = max(self.min_val, min(self.max_val, float(v)))
        self._value = float(round(v)) if self.is_int else v
        self._frac = self._v2f(self._value)
        self._changed = True

    # -- drawing --------------------------------------------------------------
    def draw(
        self,
        surf: pygame.Surface,
        font_lbl: pygame.font.Font,
        font_val: pygame.font.Font,
    ) -> None:
        x, y, w = self.rect.x, self.rect.y, self.rect.w
        lbl_col = _c("changed") if self._changed else _c("header")
        surf.blit(font_lbl.render(self.label, True, lbl_col), (x + 6, y + 4))
        vs = font_val.render(self._fmt_value(), True, _c("text"))
        surf.blit(vs, (x + w - vs.get_width() - 6, y + 4))

        # Track
        pygame.draw.rect(
            surf, _c("slider_bg"),
            (self._tx, self._ty - 3, self._tw, 6),
            border_radius=3,
        )
        fw = max(0, int(self._frac * self._tw))
        if fw:
            pygame.draw.rect(
                surf, _c("slider_fill"),
                (self._tx, self._ty - 3, fw, 6),
                border_radius=3,
            )

        # Knob
        kx = self._tx + int(self._frac * self._tw)
        pygame.draw.circle(surf, _c("slider_knob"), (kx, self._ty), 8)
        if self._drag:
            pygame.draw.circle(surf, _c("header"), (kx, self._ty), 8, 2)


# ---------------------------------------------------------------------------
# Button widget
# ---------------------------------------------------------------------------
class Button:
    """Simple rectangular button."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        label: str,
        color: Optional[Tuple[int, int, int]] = None,
        hover: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        self.rect    = pygame.Rect(rect)
        self.label   = label
        self._col    = color or _c("btn")
        self._hover  = hover or _c("btn_hover")
        self._is_hov = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if this button was clicked."""
        if event.type == pygame.MOUSEMOTION:
            self._is_hov = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(
        self,
        surf: pygame.Surface,
        font: pygame.font.Font,
        active: bool = False,
        disabled: bool = False,
    ) -> None:
        if disabled:
            col = _c("dim")
        elif self._is_hov or active:
            col = self._hover
        else:
            col = self._col
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        border_col = _c("header") if active else _c("border")
        pygame.draw.rect(surf, border_col, self.rect, 1 + int(active), border_radius=6)
        txt_col = _c("dim") if disabled else _c("text")
        txt = font.render(self.label, True, txt_col)
        surf.blit(
            txt,
            (
                self.rect.x + (self.rect.w - txt.get_width())  // 2,
                self.rect.y + (self.rect.h - txt.get_height()) // 2,
            ),
        )


# ---------------------------------------------------------------------------
# SB3 training callback
# ---------------------------------------------------------------------------
class _Callback(_BaseCallback):
    """
    Stable-Baselines3 callback that feeds live metrics to the dashboard
    and honours pause/stop events from the UI.
    """

    def __init__(
        self,
        metrics_q: LatestSlot,
        stop_ev: threading.Event,
        pause_ev: threading.Event,
        episode_q: Optional[queue.Queue] = None,
        save_req: Optional[threading.Event] = None,
        save_fn=None,  # callable(model): saves; run on the training thread
    ) -> None:
        super().__init__(verbose=0)
        self._q       = metrics_q
        self._stop    = stop_ev
        self._pause   = pause_ev
        self._episode_q = episode_q  # receives each finished episode's return
        self._save_req = save_req
        self._save_fn  = save_fn
        self._start_steps = 0  # num_timesteps when this learn() call began
        self.ep_rewards: List[float] = []
        self._avg: Optional[float] = None  # mean of the last 50 episodes
        self._algo_metrics: Dict[str, float] = {}  # latest of each logged metric
        self._ep_r    = 0.0
        self._ep_n    = 0
        self._step_n  = 0
        self._t0      = time.time()

    def _on_training_start(self) -> None:
        self._start_steps = self.num_timesteps  # > 0 when resuming a saved model

    def _save_if_requested(self) -> None:
        # Saving happens here, on the training thread between steps, so the
        # model is never written while it is being updated.
        if self._save_req is not None and self._save_req.is_set():
            self._save_req.clear()
            self._save_fn(self.model)

    def _on_step(self) -> bool:  # type: ignore[override]
        # Honour stop signal – returning False terminates model.learn()
        if self._stop.is_set():
            return False
        self._save_if_requested()

        # Honour pause – sleep until resumed or stopped (saving still works)
        while self._pause.is_set() and not self._stop.is_set():
            self._save_if_requested()
            time.sleep(0.05)
        if self._stop.is_set():
            return False

        reward = float(self.locals.get("rewards", [0])[0])
        done   = bool(self.locals.get("dones",   [False])[0])
        self._ep_r   += reward
        self._step_n += 1

        metrics: Dict = {
            "episode":     self._ep_n,
            "step":        self._step_n,
            "reward":      reward,
            "ep_reward":   self._ep_r,
            "avg_reward":  self._avg,
            # Whatever the algorithm logs, by name (e.g. DQN's
            # rollout/exploration_rate); nothing is assumed to exist.
            "algo_metrics": dict(collect_algorithm_metrics(self.logger, self._algo_metrics)),
            # Forwarded as the env reported it; the env itself is never read.
            # On an episode's last step this is still that step, although
            # the VecEnv has already reset the env.
            "render_state": self.locals["infos"][0].get("render_state"),
            "total_steps": self.num_timesteps,
            "run_steps":   self.num_timesteps - self._start_steps,
            "elapsed":     time.time() - self._t0,
        }

        # Never blocks: replaces any update the UI has not drawn yet.
        self._q.put(metrics)

        if done:
            self.ep_rewards.append(self._ep_r)
            if len(self.ep_rewards) >= 10:
                self._avg = float(np.mean(self.ep_rewards[-50:]))
            if self._episode_q is not None:
                self._episode_q.put(self._ep_r)  # once per episode, never dropped
            self._ep_r   = 0.0
            self._ep_n  += 1
            self._step_n = 0

        return True


# ---------------------------------------------------------------------------
# Background training thread
# ---------------------------------------------------------------------------
class _TrainingThread(threading.Thread):
    """Runs DQN training in a daemon thread; communicates via queues/events."""

    def __init__(
        self,
        params: dict,  # {"env_id", "env", "agent", "run" (kwargs by name), "load_path", "run_dir"}
        metrics_q: LatestSlot,
        stop_ev: threading.Event,
        pause_ev: threading.Event,
        on_done,  # callable(thread), called from this thread when it finishes
        episode_q: Optional[queue.Queue] = None,
        save_req: Optional[threading.Event] = None,
        saved_q: Optional[queue.Queue] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.params   = params
        self._episode_q = episode_q
        self._save_req = save_req or threading.Event()
        self._saved_q  = saved_q  # receives (thread, step) after each requested save
        self._q       = metrics_q
        self._stop    = stop_ev
        self._pause   = pause_ev
        self._on_done = on_done
        self.model    = None
        self._env     = None
        self.error: Optional[Exception] = None
        self.stopped  = False  # learn() returned because STOP was pressed
        self.run_dir  = Path(params["run_dir"])

    def run(self) -> None:
        try:
            p = self.params
            self.run_dir.mkdir(parents=True, exist_ok=True)
            env_config, agent_config = p["env"], p["agent"]
            seed, total_timesteps = p["run"]["seed"], p["run"]["total_timesteps"]
            write_run_config(
                self.run_dir, env_id=p["env_id"], env=env_config,
                agent={**agent_config, "exploration_fraction": EXPLORATION_FRACTION,
                       "learning_starts": LEARNING_STARTS,
                       "target_update_interval": TARGET_UPDATE_INTERVAL},
                seed=seed, total_timesteps=total_timesteps,
                loaded_from=p.get("load_path"),
            )
            # Same construction as train_agent.py, so identical settings
            # train identically from either entry point.
            env = make_env(p["env_id"], seed=seed, monitor_dir=self.run_dir, **env_config)
            self._env = env

            load_path = p.get("load_path")
            if load_path:
                # Resume the saved model: its weights, optimizer state,
                # hyperparameters, step count and replay buffer. The agent
                # sliders do not apply to a resumed run.
                model = load_run_model(load_path, env, seed=seed)
            else:
                model = make_dqn(env, **agent_config, seed=seed, verbose=0)
            attach_run_logger(model, self.run_dir)

            self.model = model
            cb = _Callback(self._q, self._stop, self._pause, self._episode_q,
                           save_req=self._save_req, save_fn=self._save)
            model.learn(
                total_timesteps=total_timesteps,
                callback=CallbackList([cb, checkpoint_callback(self.run_dir)]),
                reset_num_timesteps=not load_path,
            )
            self.stopped = self._stop.is_set()
            # A stopped run is partial: never let it overwrite the saved model.
            # (It stays available to SAVE MODEL explicitly.)
            if not self.stopped:
                save_run_model(model, self.run_dir / "model")

        except Exception as exc:  # noqa: BLE001
            self.error = exc
        finally:
            if self._env is not None:
                self._env.close()
            self._on_done(self)

    def _save(self, model) -> None:
        """Save on request from the UI. Runs on this thread, between steps."""
        save_run_model(model, self.run_dir / "model")
        if self._saved_q is not None:
            self._saved_q.put((self, model.num_timesteps))


# ---------------------------------------------------------------------------
# Dashboard (main class)
# ---------------------------------------------------------------------------
class Dashboard:
    """
    Top-level PyGame UI.  Builds sliders + buttons, drives the render loop,
    and manages the background training thread.
    """

    def __init__(
        self,
        env_id: str = WAREHOUSE_ID,
        renderer: Optional[Renderer] = None,
        env_params: Optional[Sequence[Param]] = None,
    ) -> None:
        """Train ``env_id``, any registered Gymnasium id. Its profile
        (lab_envs.py) supplies the renderer and environment sliders unless
        ``renderer`` / ``env_params`` are given."""
        if not _SB3_AVAILABLE:
            raise SystemExit(
                "stable-baselines3 is required.  "
                "Install it with:  pip install stable-baselines3"
            )

        self._env_id  = env_id
        self._profile = get_profile(env_id)
        # Draws the environment in the centre panel (see lab_render.py)
        self._renderer = renderer if renderer is not None else self._profile.renderer()
        if env_params is None:
            env_params = self._profile.params
        # One slider per setting, in this order (see lab_params.py)
        self._params: Tuple[Param, ...] = (*env_params, *AGENT_PARAMS, *RUN_PARAMS)

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(f"RL Training Lab - {env_id}")
        self.clock = pygame.time.Clock()

        # Font set
        self._ft = pygame.font.SysFont("monospace", 16, bold=True)   # title
        self._fl = pygame.font.SysFont("monospace", 12)               # label
        self._fv = pygame.font.SysFont("monospace", 12, bold=True)    # value
        self._fb = pygame.font.SysFont("monospace", 13, bold=True)    # button
        self._fs = pygame.font.SysFont("monospace", 11)               # small

        self._build_ui()

        # Application state
        self._state     = "idle"   # idle | training | paused | done | error
        self._model     = None
        self._model_status = "No model"
        self._load_path: Optional[str] = None
        self._run_dir: Optional[Path] = None  # current run's output folder
        self._metrics: Dict = {}
        self._ep_rewards: List[float] = []
        self._fps       = 0
        self._err_msg   = ""

        # Threading primitives
        # Newest metrics from the training thread, and every finished
        # episode's return; new ones per run.
        self._metrics_slot = LatestSlot()
        self._episode_q: queue.Queue = queue.Queue()
        # SAVE while training: the UI sets the event, the training thread
        # saves at its next step and reports back on the queue.
        self._save_req = threading.Event()
        self._saved_q: queue.Queue = queue.Queue()
        # Finished training threads, posted by the thread itself and applied
        # by the UI thread, which is the only writer of the dashboard state.
        self._done_q: "queue.Queue[_TrainingThread]" = queue.Queue()
        self._stop_ev    = threading.Event()
        self._pause_ev   = threading.Event()
        self._thread: Optional[_TrainingThread] = None

        # FPS counter
        self._frame_n   = 0
        self._fps_timer = time.time()

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Sliders (left panel)
        sx, sw, sy = 12, LEFT_W - 24, 52
        gap = 48
        self._sliders: Dict[str, Slider] = {
            p.name: Slider(sx, sy + i * gap, sw, p.label, p.min, p.max, p.default,
                           p.type is int, p.log, p.fmt)
            for i, p in enumerate(self._params)
        }

        # Buttons (bottom bar)  –  seven buttons spread across the full width
        by  = CENTER_H + 20
        bh  = 42
        bw  = 155
        gap_b = (SCREEN_W - 32 - 7 * bw) // 6
        bx  = 16
        specs = [
            ("START", "START TRAINING", _c("btn_start")),
            ("PAUSE", "PAUSE",          _c("btn")),
            ("STOP",  "STOP",           _c("btn_stop")),
            ("SAVE",  "SAVE MODEL",     _c("btn_save")),
            ("LOAD",  "LOAD MODEL",     _c("btn")),
            ("RESET", "RESET STATS",    _c("btn")),
            ("EXIT",  "EXIT",           _c("btn_stop")),
        ]
        self._btns: Dict[str, Button] = {
            key: Button((bx + i * (bw + gap_b), by, bw, bh), lbl, color=col)
            for i, (key, lbl, col) in enumerate(specs)
        }

    # -------------------------------------------------------------------------
    # Main render / event loop
    # -------------------------------------------------------------------------
    def run(self) -> None:
        running = True
        while running:
            self.clock.tick(FPS_CAP)

            # FPS counter
            self._frame_n += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self._fps       = self._frame_n
                self._frame_n   = 0
                self._fps_timer = now

            # Newest metrics from the training thread, if any arrived
            has_update, m = self._metrics_slot.take()
            if has_update:
                self._metrics = m
            self._drain_episodes()
            self._drain_saved()

            # Apply the result of a training thread that has finished
            self._apply_finished()

            # ── Event handling ───────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    self._on_key(event.key)

                for s in self._sliders.values():
                    s.handle_event(event)

                if self._btns["START"].handle_event(event):
                    self._cmd_start()
                if self._btns["PAUSE"].handle_event(event):
                    self._cmd_pause()
                if self._btns["STOP"].handle_event(event):
                    self._cmd_stop()
                if self._btns["SAVE"].handle_event(event):
                    self._cmd_save()
                if self._btns["LOAD"].handle_event(event):
                    self._cmd_load()
                if self._btns["RESET"].handle_event(event):
                    self._cmd_reset()
                if self._btns["EXIT"].handle_event(event):
                    running = False

            # ── Render ───────────────────────────────────────────────────────
            self.screen.fill(_c("bg"))
            self._draw_left_panel()
            self._draw_center()
            self._draw_right_panel()
            self._draw_bottom_bar()
            pygame.display.flip()

        self._cmd_stop()
        pygame.quit()

    # -------------------------------------------------------------------------
    # Key handler
    # -------------------------------------------------------------------------
    def _on_key(self, key: int) -> None:
        if key == pygame.K_SPACE:
            self._cmd_pause()
        elif key in (pygame.K_q, pygame.K_ESCAPE):
            self._cmd_stop()
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif key == pygame.K_s:
            self._cmd_save()
        elif key == pygame.K_r:
            self._cmd_reset()

    # -------------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------------
    def _get_params(self) -> dict:
        """Slider values by target and name: {"env": {...}, "agent": {...}, "run": {...}}."""
        values = {p.name: p.value(self._sliders[p.name].value) for p in self._params}
        return {
            "env_id":    self._env_id,
            **by_target(self._params, values),
            "load_path": self._load_path,
            "run_dir":   None,  # set by _cmd_start
        }

    def _cmd_start(self) -> None:
        if self._state in ("training", "paused"):
            return
        if self._thread and self._thread.is_alive():
            self._model_status = "Previous run still stopping"
            return
        if not _SB3_AVAILABLE:
            self._model_status = "Missing dependencies – cannot train"
            return
        self._stop_ev.clear()
        self._pause_ev.clear()
        params = self._get_params()
        self._run_dir = new_run_dir(
            RUNS_DIR, self._profile.run_tag(params["env"]), params["run"]["seed"]
        )
        params["run_dir"] = str(self._run_dir)
        for s in self._sliders.values():
            s.mark_applied()
        self._ep_rewards = []
        self._metrics    = {}
        self._err_msg    = ""
        self._btns["PAUSE"].label = "PAUSE"
        self._metrics_slot = LatestSlot()
        self._episode_q = queue.Queue()
        self._save_req.clear()
        self._thread = _TrainingThread(
            params, self._metrics_slot, self._stop_ev, self._pause_ev,
            on_done=self._on_done, episode_q=self._episode_q,
            save_req=self._save_req, saved_q=self._saved_q,
        )
        if self._load_path:
            self._model_status = f"Resuming {Path(self._load_path).parent.name}"
            self._load_path = None  # one resume per LOAD
        self._state = "training"
        self._thread.start()

    def _cmd_pause(self) -> None:
        if self._state == "training":
            self._pause_ev.set()
            self._state = "paused"
            self._btns["PAUSE"].label = "RESUME"
        elif self._state == "paused":
            self._pause_ev.clear()
            self._state = "training"
            self._btns["PAUSE"].label = "PAUSE"

    def _cmd_stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop_ev.set()
            self._pause_ev.clear()     # ensure training loop can exit
            self._thread.join(timeout=6)
        self._apply_finished()
        if self._state in ("training", "paused"):
            # Did not finish within the timeout; its result is applied
            # (as stopped) whenever it arrives.
            self._state = "idle"
        self._btns["PAUSE"].label = "PAUSE"

    def _cmd_save(self) -> None:
        if self._state in ("training", "paused"):
            # The training thread saves at its next step (also while paused).
            self._save_req.set()
            self._model_status = "Saving..."
        elif self._model is not None:
            save_run_model(self._model, self._run_dir / "model")
            self._model_status = "Saved (run folder)"
        else:
            self._model_status = "No model to save yet"

    def _drain_saved(self) -> None:
        """Report saves the training thread made on request."""
        while True:
            try:
                thread, step = self._saved_q.get_nowait()
            except queue.Empty:
                return
            if thread is self._thread:
                self._model_status = f"Saved at step {step:,}"

    def _cmd_load(self) -> None:
        if self._state in ("training", "paused"):
            return
        latest = find_latest_model(RUNS_DIR)
        if latest is None:
            self._model_status = f"No saved model in {RUNS_DIR}/"
            return
        # Match the environment sliders to the saved run, so its model fits.
        config = read_run_config(latest.parent)
        # Runs saved before config.json recorded env_id were all warehouse runs.
        saved_id = config.get("env_id", WAREHOUSE_ID)
        if saved_id != self._env_id:
            self._model_status = f"Can't load: run is for {saved_id}"
            return
        env = config.get("env", {})
        env_params = [p for p in self._params if p.target == "env" and p.name in env]
        for p in env_params:
            if not p.in_range(env[p.name]):
                self._model_status = f"Can't load: {p.label} {env[p.name]} out of range"
                return
        for p in env_params:
            self._sliders[p.name].set_value(p.shown(env[p.name]))
        self._load_path = str(latest.with_suffix(""))
        self._model_status = f"Selected {latest.parent.name} (on START)"

    def _drain_episodes(self) -> None:
        """Append newly finished episode returns to the graph's history."""
        while True:
            try:
                self._ep_rewards.append(self._episode_q.get_nowait())
            except queue.Empty:
                break
        del self._ep_rewards[:-GRAPH_EPISODES]

    def _cmd_reset(self) -> None:
        self._ep_rewards = []
        self._metrics    = {}

    def _on_done(self, thread: "_TrainingThread") -> None:
        """Called from the training thread when it finishes; only posts a message."""
        self._done_q.put(thread)

    def _apply_finished(self) -> None:
        """Apply finished-thread results.  UI thread only."""
        while True:
            try:
                thread = self._done_q.get_nowait()
            except queue.Empty:
                return
            if thread is not self._thread:
                continue  # result of an earlier run; a newer one has started
            self._model = thread.model
            if thread.error:
                self._state        = "error"
                self._err_msg      = str(thread.error)
                self._model_status = "Error during training"
            elif thread.stopped:
                self._state        = "idle"
                self._model_status = "Stopped (unsaved)"
            else:
                self._state        = "done"
                self._model_status = "Saved (run folder)"

    # -------------------------------------------------------------------------
    # Drawing helpers
    # -------------------------------------------------------------------------
    def _blit(
        self,
        surf: pygame.Surface,
        font: pygame.font.Font,
        text: str,
        x: int,
        y: int,
        color: str = "text",
    ) -> int:
        """Render text and return the new y (below the rendered line)."""
        s = font.render(text, True, _c(color))
        surf.blit(s, (x, y))
        return y + s.get_height() + 3

    def _hdr(
        self,
        surf: pygame.Surface,
        text: str,
        x: int, y: int,
        right_x: int,
    ) -> int:
        """Section header with underline; returns next y."""
        surf.blit(self._fv.render(text, True, _c("header")), (x, y))
        y += 16
        pygame.draw.line(surf, _c("border"), (x, y), (right_x, y), 1)
        return y + 5

    # -------------------------------------------------------------------------
    # Left panel: parameter sliders
    # -------------------------------------------------------------------------
    def _draw_left_panel(self) -> None:
        surf = self.screen
        pygame.draw.rect(surf, _c("panel"), (0, 0, LEFT_W, CENTER_H))
        pygame.draw.line(surf, _c("border"), (LEFT_W - 1, 0), (LEFT_W - 1, CENTER_H), 1)

        # Title
        t = self._ft.render("== PARAMETERS ==", True, _c("header"))
        surf.blit(t, (LEFT_W // 2 - t.get_width() // 2, 14))
        pygame.draw.line(surf, _c("border"), (8, 40), (LEFT_W - 8, 40), 1)

        training_active = self._state in ("training", "paused")
        for s in self._sliders.values():
            s.draw(surf, self._fl, self._fv)
            if training_active:
                # Dim sliders slightly while training to signal they're locked
                dim = pygame.Surface((s.rect.w, s.rect.h), pygame.SRCALPHA)
                dim.fill((0, 0, 0, 70))
                surf.blit(dim, s.rect.topleft)

        # Unsaved-change indicator
        if any(s._changed for s in self._sliders.values()):
            note = self._fs.render(
                "* Param changed -- restart to apply", True, _c("changed")
            )
            y = max(s.rect.bottom for s in self._sliders.values()) + 8
            surf.blit(note, (8, y))

    # -------------------------------------------------------------------------
    # Center: the environment, drawn by its renderer
    # -------------------------------------------------------------------------
    def _draw_center(self) -> None:
        surf = self.screen
        m    = self._metrics
        area = pygame.Rect(LEFT_W + 20, 20, CENTER_W - 40, CENTER_H - 60)
        # The render state as the env pushed it; None before the first update.
        drawn = self._renderer.draw(surf, area, m.get("render_state")) or area
        ox, oy, gw, gh = drawn

        # Episode / step / reward overlay below the drawing
        ep_r = m.get("ep_reward", 0.0)
        ep   = m.get("episode",   0)
        step = m.get("step",      0)
        row_y = oy + gh + 10
        tx    = ox
        for txt, col in [
            (f"Episode {ep}",      "text"),
            (f"  Step {step}",     "text"),
            (f"  Reward {ep_r:+.2f}", "pos" if ep_r >= 0 else "neg"),
        ]:
            s = self._fl.render(txt, True, _c(col))
            surf.blit(s, (tx, row_y))
            tx += s.get_width() + 6

        # PAUSED banner
        if self._state == "paused":
            banner_h = 38
            ban = pygame.Surface((gw + 20, banner_h), pygame.SRCALPHA)
            ban.fill((*_c("paused_bg"), 210))
            surf.blit(ban, (ox - 10, oy))
            pt = self._fb.render(
                "  [PAUSED]  SPACE to resume  ", True, _c("paused_text")
            )
            surf.blit(pt, (ox + (gw - pt.get_width()) // 2, oy + (banner_h - pt.get_height()) // 2))

        # Idle / done / error message
        if self._state in ("idle", "done", "error") and not m:
            msgs = {
                "idle":  "Configure parameters and press  START TRAINING",
                "done":  "Training complete!  Press SAVE MODEL or START again.",
                "error": f"Error: {self._err_msg}",
            }
            col = "neg" if self._state == "error" else "dim"
            ms  = self._fl.render(msgs[self._state], True, _c(col))
            surf.blit(ms, (ox + (gw - ms.get_width()) // 2, oy + gh // 2 - ms.get_height() // 2))

    # -------------------------------------------------------------------------
    # Right panel: stats + reward graph
    # -------------------------------------------------------------------------
    def _draw_right_panel(self) -> None:
        surf = self.screen
        rx   = LEFT_W + CENTER_W
        pygame.draw.rect(surf, _c("panel"), (rx, 0, RIGHT_W, CENTER_H))
        pygame.draw.line(surf, _c("border"), (rx, 0), (rx, CENTER_H), 1)

        m   = self._metrics
        x   = rx + 12
        y   = 14

        def hdr(txt: str) -> None:
            nonlocal y
            y = self._hdr(surf, txt, x, y, rx + RIGHT_W - 12)

        def row(lbl: str, val: str, col: str = "text") -> None:
            nonlocal y
            ls = self._fl.render(lbl, True, _c("dim"))
            vs = self._fl.render(val,  True, _c(col))
            surf.blit(ls, (x, y))
            surf.blit(vs, (rx + RIGHT_W - vs.get_width() - 12, y))
            y += ls.get_height() + 3

        # -- Training status --------------------------------------------------
        hdr("=== STATUS ===")
        state_col = {
            "idle":     "dim",
            "training": "pos",
            "paused":   "warn",
            "done":     "header",
            "error":    "neg",
        }.get(self._state, "text")
        row("State:", self._state.upper(), col=state_col)
        ms = self._model_status
        if len(ms) > 27:
            ms = ms[:24] + "..."
        row("Model:", ms)
        if self._run_dir is not None:
            row("Run:", self._run_dir.name[-27:])
        y += 6

        # -- Episode metrics --------------------------------------------------
        hdr("=== EPISODE ===")
        ep_r = m.get("ep_reward", 0.0)
        row("Episode #:", str(m.get("episode", 0)))
        row("Step:",      str(m.get("step", 0)))
        row("Ep Reward:", f"{ep_r:+.2f}", col="pos" if ep_r >= 0 else "neg")
        avg = m.get("avg_reward")
        row(
            "Avg(50ep):",
            f"{avg:+.2f}" if avg is not None else "---",
            col="pos" if (avg or 0) >= 0 else "neg",
        )
        y += 6

        # -- Algorithm metrics, whichever the algorithm logs -----------------
        hdr("=== ALGORITHM ===")
        shown = display_metrics(m.get("algo_metrics") or {}, ALGO_ROWS)
        for name, value in shown:
            row(f"{name}:", value)
        if not shown:
            row("(none logged yet)", "")
        y += 6

        # -- Performance ------------------------------------------------------
        hdr("=== PERFORMANCE ===")
        ts = m.get("total_steps", 0)
        el = m.get("elapsed", 0.0)
        mm, ss = divmod(int(el), 60)
        hh, mm = divmod(mm, 60)
        row("Total Steps:", f"{ts:,}")
        row("Elapsed:",     f"{hh:02d}:{mm:02d}:{ss:02d}")
        row("UI FPS:",      str(self._fps))
        # Steps whose state was replaced before a frame could draw it
        row("Not drawn:",   f"{self._metrics_slot.overwritten:,}")

        # Progress bar when training
        if self._state in ("training", "paused") and self._thread:
            total = self._thread.params["run"]["total_timesteps"]
            pct   = min(1.0, m.get("run_steps", 0) / max(total, 1))
            bar_w = RIGHT_W - 24
            bar_h = 8
            bar_x, bar_y = x, y + 2
            pygame.draw.rect(surf, _c("slider_bg"),   (bar_x, bar_y, bar_w, bar_h), border_radius=4)
            fw = max(0, int(pct * bar_w))
            if fw:
                pygame.draw.rect(surf, _c("slider_fill"), (bar_x, bar_y, fw, bar_h), border_radius=4)
            pct_s = self._fs.render(f"{int(pct * 100)}%", True, _c("dim"))
            surf.blit(pct_s, (bar_x + bar_w - pct_s.get_width(), bar_y + bar_h + 2))
            y += 22

        y += 6

        # -- Reward history graph ---------------------------------------------
        hdr("=== REWARD HISTORY ===")
        if len(self._ep_rewards) >= 2:
            gh  = 140
            gw  = RIGHT_W - 24
            self._draw_graph(surf, x, y, gw, gh, self._ep_rewards)
            y  += gh + 18
        else:
            y = self._blit(surf, self._fs, "Waiting for episodes...", x, y, color="dim")
            y += 4

        # -- Keyboard shortcuts -----------------------------------------------
        y = max(y, CENTER_H - 80)
        hdr("=== SHORTCUTS ===")
        for k, v in [
            ("SPACE", "pause / resume"),
            ("S",     "save model"),
            ("R",     "reset stats"),
            ("Q/ESC", "quit"),
        ]:
            y = self._blit(surf, self._fs, f"{k:<7}  {v}", x, y, color="dim")

    def _draw_graph(
        self,
        surf: pygame.Surface,
        gx: int, gy: int, gw: int, gh: int,
        rewards: List[float],
    ) -> None:
        """Draw episode reward history and 20-ep moving average."""
        pygame.draw.rect(surf, _c("graph_bg"), (gx, gy, gw, gh))
        pygame.draw.rect(surf, _c("border"),   (gx, gy, gw, gh), 1)

        mn, mx = min(rewards), max(rewards)
        rng = (mx - mn) if mx != mn else 1.0

        # Horizontal grid lines
        for frac in (0.25, 0.5, 0.75):
            py = gy + int(gh * (1 - frac))
            pygame.draw.line(surf, _c("graph_grid"), (gx, py), (gx + gw, py), 1)

        def to_px(i: int, v: float) -> Tuple[int, int]:
            px = gx + 1 + int(i / max(len(rewards) - 1, 1) * (gw - 2))
            py = gy + gh - 1 - int((v - mn) / rng * (gh - 2))
            return px, py

        # Raw episode rewards (thin blue line)
        pts = [to_px(i, v) for i, v in enumerate(rewards)]
        if len(pts) >= 2:
            pygame.draw.lines(surf, _c("graph_raw"), False, pts, 1)

        # 20-episode moving average (thicker orange line)
        win = 20
        if len(rewards) >= win:
            avg_pts = [
                to_px(i, float(np.mean(rewards[i - win + 1: i + 1])))
                for i in range(win - 1, len(rewards))
            ]
            if len(avg_pts) >= 2:
                pygame.draw.lines(surf, _c("graph_avg"), False, avg_pts, 2)

        # Axis labels
        mn_s = self._fs.render(f"{mn:.1f}", True, _c("neg"))
        mx_s = self._fs.render(f"{mx:.1f}", True, _c("pos"))
        ep_s = self._fs.render(f"ep {len(rewards)}", True, _c("dim"))
        surf.blit(mn_s, (gx + 2, gy + gh - mn_s.get_height() - 1))
        surf.blit(mx_s, (gx + 2, gy + 1))
        surf.blit(ep_s, (gx + gw - ep_s.get_width() - 2, gy + gh - ep_s.get_height() - 1))

        # Legend
        ly = gy + gh + 3
        pygame.draw.line(surf, _c("graph_raw"), (gx, ly + 5), (gx + 14, ly + 5), 1)
        surf.blit(self._fs.render("raw", True, _c("graph_raw")), (gx + 16, ly))
        pygame.draw.line(surf, _c("graph_avg"), (gx + 52, ly + 5), (gx + 66, ly + 5), 2)
        surf.blit(self._fs.render("avg20", True, _c("graph_avg")), (gx + 68, ly))

    # -------------------------------------------------------------------------
    # Bottom bar: control buttons + status line
    # -------------------------------------------------------------------------
    def _draw_bottom_bar(self) -> None:
        surf = self.screen
        pygame.draw.rect(surf, _c("panel"),  (0, CENTER_H, SCREEN_W, BOTTOM_H))
        pygame.draw.line(surf, _c("border"), (0, CENTER_H), (SCREEN_W, CENTER_H), 1)

        training = self._state in ("training", "paused")
        idle     = self._state in ("idle", "done", "error")

        self._btns["START"].draw(surf, self._fb, disabled=training)
        self._btns["PAUSE"].draw(
            surf, self._fb,
            active=self._state == "paused",
            disabled=not training,
        )
        self._btns["STOP"].draw( surf, self._fb, disabled=idle)
        self._btns["SAVE"].draw( surf, self._fb, disabled=self._model is None and not training)
        self._btns["LOAD"].draw( surf, self._fb, disabled=training)
        self._btns["RESET"].draw(surf, self._fb)
        self._btns["EXIT"].draw( surf, self._fb)

        # Status line centred along the bottom edge
        parts = []
        if training and self._thread:
            ts    = self._metrics.get("run_steps", 0)
            total = self._thread.params["run"]["total_timesteps"]
            pct   = min(100, int(ts / max(total, 1) * 100))
            parts.append(f"Progress: {pct}%  ({ts:,} steps)")
        parts.append(f"FPS: {self._fps}")
        if self._state == "error":
            parts.append(f"ERR: {self._err_msg[:40]}")
        status = "   |   ".join(parts)
        ss = self._fs.render(status, True, _c("dim"))
        surf.blit(
            ss,
            (
                SCREEN_W // 2 - ss.get_width() // 2,
                CENTER_H + BOTTOM_H - ss.get_height() - 8,
            ),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Interactive RL training lab")
    parser.add_argument(
        "--env", default=WAREHOUSE_ID,
        help=f"Gymnasium environment id to train (default: {WAREHOUSE_ID}); "
             "'module:Env-v0' imports module first",
    )
    Dashboard(env_id=parser.parse_args().env).run()


if __name__ == "__main__":
    main()
