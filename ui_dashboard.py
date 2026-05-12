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

import functools
import math
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
except ImportError as exc:
    raise SystemExit(
        "pygame is required.  Install it with:  pip install pygame"
    ) from exc

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
    _SB3_AVAILABLE = True
except ImportError:
    DQN = None          # type: ignore[assignment,misc]
    make_vec_env = None  # type: ignore[assignment]
    _SB3_AVAILABLE = False

    class _BaseCallback:  # type: ignore[no-redef]
        """Stub so the _Callback class definition does not crash at import."""
        def __init__(self, verbose: int = 0) -> None:
            pass

try:
    from warehouse_env import WarehouseEnv
    _ENV_AVAILABLE = True
except ImportError:
    WarehouseEnv = None  # type: ignore[assignment,misc]
    _ENV_AVAILABLE = False


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
    "grid_line":    ( 50,  57,  84),
    "delivery":     ( 50, 135,  58),
    "delivery_lbl": (120, 215, 120),
    "agent":        (218, 218, 225),
    "agent_carry":  (255, 200,  48),
    "pkg_pend":     ( 62, 108, 208),
    "pkg_done":     ( 62, 182,  80),
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
        metrics_q: queue.Queue,
        stop_ev: threading.Event,
        pause_ev: threading.Event,
    ) -> None:
        super().__init__(verbose=0)
        self._q       = metrics_q
        self._stop    = stop_ev
        self._pause   = pause_ev
        self.ep_rewards: List[float] = []
        self._ep_r    = 0.0
        self._ep_n    = 0
        self._step_n  = 0
        self._t0      = time.time()

    def _on_step(self) -> bool:  # type: ignore[override]
        # Honour stop signal – returning False terminates model.learn()
        if self._stop.is_set():
            return False

        # Honour pause – sleep until resumed or stopped
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.05)
        if self._stop.is_set():
            return False

        reward = float(self.locals.get("rewards", [0])[0])
        done   = bool(self.locals.get("dones",   [False])[0])
        self._ep_r   += reward
        self._step_n += 1

        avg: Optional[float] = (
            float(np.mean(self.ep_rewards[-50:]))
            if len(self.ep_rewards) >= 10 else None
        )
        eps = float(getattr(self.model, "exploration_rate", 1.0))

        # Unwrap VecEnv / Monitor wrappers to reach the raw WarehouseEnv
        raw = self.training_env.envs[0]
        while hasattr(raw, "env"):
            raw = raw.env

        metrics: Dict = {
            "episode":     self._ep_n,
            "step":        self._step_n,
            "reward":      reward,
            "ep_reward":   self._ep_r,
            "avg_reward":  avg,
            "epsilon":     eps,
            "agent_pos":   raw.agent_pos.tolist(),
            "packages":    [list(p) for p in raw.packages],
            "carrying":    int(raw.carrying),
            "grid_size":   int(raw.grid_size),
            "total_steps": self.num_timesteps,
            "elapsed":     time.time() - self._t0,
            "ep_rewards":  list(self.ep_rewards[-100:]),
        }

        try:
            self._q.put_nowait(metrics)
        except queue.Full:
            pass  # drop frame; keep training at full speed

        if done:
            self.ep_rewards.append(self._ep_r)
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
        params: dict,
        metrics_q: queue.Queue,
        stop_ev: threading.Event,
        pause_ev: threading.Event,
        on_done,  # callable(model, error)
    ) -> None:
        super().__init__(daemon=True)
        self.params   = params
        self._q       = metrics_q
        self._stop    = stop_ev
        self._pause   = pause_ev
        self._on_done = on_done
        self.model    = None
        self.error: Optional[Exception] = None

    def run(self) -> None:
        try:
            p = self.params
            make_fn = functools.partial(
                WarehouseEnv,
                grid_size    = int(p["grid_size"]),
                num_packages = int(p["num_packages"]),
                max_steps    = int(p["max_steps"]),
                step_cost    = float(p["step_cost"]),
            )
            env = make_vec_env(make_fn, n_envs=1)

            model = DQN(
                "MlpPolicy",
                env,
                learning_rate           = float(p["learning_rate"]),
                buffer_size             = int(p["buffer_size"]),
                learning_starts         = min(500, int(p["total_timesteps"]) // 20),
                target_update_interval  = 500,
                exploration_fraction    = 0.3,
                exploration_initial_eps = float(p["exploration_initial"]),
                exploration_final_eps   = float(p["exploration_final"]),
                verbose                 = 0,
            )

            # Continue from an existing saved model if requested
            load_path = p.get("load_path")
            if load_path and os.path.exists(load_path + ".zip"):
                loaded = DQN.load(load_path, env=env)
                model.set_parameters(loaded.get_parameters())

            self.model = model
            cb = _Callback(self._q, self._stop, self._pause)
            model.learn(total_timesteps=int(p["total_timesteps"]), callback=cb)
            model.save("warehouse_dqn_model")

        except Exception as exc:  # noqa: BLE001
            self.error = exc
        finally:
            self._on_done(self.model, self.error)


# ---------------------------------------------------------------------------
# Dashboard (main class)
# ---------------------------------------------------------------------------
class Dashboard:
    """
    Top-level PyGame UI.  Builds sliders + buttons, drives the render loop,
    and manages the background training thread.
    """

    # (label, min, max, default, is_int, log_scale, fmt)
    _SLIDER_SPECS: List[Tuple] = [
        ("Grid Size",       6,    15,    8,    True,  False, None),
        ("Packages",        1,     5,    1,    True,  False, None),
        ("Max Steps",     100,   500,  300,    True,  False, None),
        ("Timesteps (k)", 10,   500,  100,    True,  False, None),
        ("Learning Rate", 1e-4, 1e-2, 1e-3,  False,  True,  "{:.2e}"),
        ("Init Epsilon",  0.5,   1.0,  1.0,  False,  False, None),
        ("Final Epsilon", 0.01,  0.3,  0.1,  False,  False, None),
        ("Buffer (k)",      1,    50,   10,   True,  False, None),
        ("Step Cost",    0.01,   0.5, 0.05,  False,  False, None),
    ]

    def __init__(self) -> None:
        if not _SB3_AVAILABLE:
            raise SystemExit(
                "stable-baselines3 is required.  "
                "Install it with:  pip install stable-baselines3"
            )
        if not _ENV_AVAILABLE:
            raise SystemExit(
                "warehouse_env.py not found.  "
                "Run this script from the project directory."
            )

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Warehouse RL Training Lab")
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
        self._metrics: Dict = {}
        self._ep_rewards: List[float] = []
        self._fps       = 0
        self._err_msg   = ""

        # Threading primitives
        self._metrics_q  = queue.Queue(maxsize=300)
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
        self._sliders: List[Slider] = [
            Slider(sx, sy + i * gap, sw, lbl, mn, mx, df, ii, ls, fm)
            for i, (lbl, mn, mx, df, ii, ls, fm) in enumerate(self._SLIDER_SPECS)
        ]

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

            # Drain the metrics queue from the training thread
            try:
                while True:
                    m = self._metrics_q.get_nowait()
                    self._metrics = m
                    ep = m.get("ep_rewards")
                    if ep is not None:
                        self._ep_rewards = ep
            except queue.Empty:
                pass

            # Detect training thread finishing unexpectedly
            if self._state in ("training", "paused"):
                if self._thread and not self._thread.is_alive():
                    if self._state not in ("done", "error", "idle"):
                        self._state = "done"

            # ── Event handling ───────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    self._on_key(event.key)

                for s in self._sliders:
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
        s = self._sliders
        return {
            "grid_size":           int(s[0].value),
            "num_packages":        int(s[1].value),
            "max_steps":           int(s[2].value),
            "total_timesteps":     int(s[3].value) * 1000,
            "learning_rate":       s[4].value,
            "exploration_initial": s[5].value,
            "exploration_final":   s[6].value,
            "buffer_size":         int(s[7].value) * 1000,
            "step_cost":           s[8].value,
            "load_path":           self._load_path,
        }

    def _cmd_start(self) -> None:
        if self._state in ("training", "paused"):
            return
        if not _SB3_AVAILABLE or not _ENV_AVAILABLE:
            self._model_status = "Missing dependencies – cannot train"
            return
        self._stop_ev.clear()
        self._pause_ev.clear()
        params = self._get_params()
        for s in self._sliders:
            s.mark_applied()
        self._ep_rewards = []
        self._metrics    = {}
        self._err_msg    = ""
        self._btns["PAUSE"].label = "PAUSE"
        self._thread = _TrainingThread(
            params, self._metrics_q, self._stop_ev, self._pause_ev,
            on_done=self._on_done,
        )
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
        self._state = "idle"
        self._btns["PAUSE"].label = "PAUSE"

    def _cmd_save(self) -> None:
        if self._model is not None:
            path = "warehouse_dqn_model"
            self._model.save(path)
            self._model_status = f"Saved: {path}.zip"
        else:
            self._model_status = "No model to save yet"

    def _cmd_load(self) -> None:
        path = "warehouse_dqn_model"
        if os.path.exists(path + ".zip"):
            self._load_path    = path
            self._model_status = f"Loaded: {path}.zip (applies on next START)"
        else:
            self._model_status = "File not found: warehouse_dqn_model.zip"

    def _cmd_reset(self) -> None:
        self._ep_rewards = []
        self._metrics    = {}

    def _on_done(self, model, error: Optional[Exception]) -> None:
        """Called from the training thread when training finishes."""
        self._model = model
        if error:
            self._state        = "error"
            self._err_msg      = str(error)
            self._model_status = f"Error during training"
        else:
            self._state        = "done"
            self._model_status = "Trained (unsaved)"

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
        for s in self._sliders:
            s.draw(surf, self._fl, self._fv)
            if training_active:
                # Dim sliders slightly while training to signal they're locked
                dim = pygame.Surface((s.rect.w, s.rect.h), pygame.SRCALPHA)
                dim.fill((0, 0, 0, 70))
                surf.blit(dim, s.rect.topleft)

        # Unsaved-change indicator
        if any(s._changed for s in self._sliders):
            note = self._fs.render(
                "* Param changed -- restart to apply", True, _c("changed")
            )
            y = self._sliders[-1].rect.bottom + 8
            surf.blit(note, (8, y))

    # -------------------------------------------------------------------------
    # Center: live warehouse grid
    # -------------------------------------------------------------------------
    def _draw_center(self) -> None:
        surf = self.screen
        m    = self._metrics
        gs   = int(m.get("grid_size", int(self._sliders[0].value)))

        # Dynamic cell size to fill the available area
        cell = min((CENTER_W - 40) // max(gs, 1), (CENTER_H - 60) // max(gs, 1))
        cell = max(16, cell)
        gw   = cell * gs
        gh   = cell * gs
        ox   = LEFT_W + (CENTER_W - gw) // 2
        oy   = (CENTER_H - gh) // 2 - 10

        # Delivery zone (bottom-right cell)
        dz = gs - 1
        pygame.draw.rect(surf, _c("delivery"), (ox + dz * cell, oy + dz * cell, cell, cell))
        dl = self._fs.render("D", True, _c("delivery_lbl"))
        surf.blit(
            dl,
            (
                ox + dz * cell + (cell - dl.get_width())  // 2,
                oy + dz * cell + (cell - dl.get_height()) // 2,
            ),
        )

        # Grid lines
        for c in range(gs + 1):
            x = ox + c * cell
            pygame.draw.line(surf, _c("grid_line"), (x, oy), (x, oy + gh))
        for r in range(gs + 1):
            y = oy + r * cell
            pygame.draw.line(surf, _c("grid_line"), (ox, y), (ox + gw, y))

        # Packages
        for pkg in m.get("packages", []):
            px, py, delivered = int(pkg[0]), int(pkg[1]), bool(pkg[2])
            col = _c("pkg_done") if delivered else _c("pkg_pend")
            mg  = max(3, cell // 6)
            pygame.draw.rect(
                surf, col,
                (ox + px * cell + mg, oy + py * cell + mg, cell - 2 * mg, cell - 2 * mg),
                border_radius=max(2, cell // 8),
            )

        # Agent
        ap      = m.get("agent_pos", [0, 0])
        carrying = m.get("carrying", -1)
        ax, ay  = int(ap[0]), int(ap[1])
        acol    = _c("agent_carry") if carrying >= 0 else _c("agent")
        rad     = max(4, cell // 3)
        cx, cy  = ox + ax * cell + cell // 2, oy + ay * cell + cell // 2
        pygame.draw.circle(surf, acol, (cx, cy), rad)
        pygame.draw.circle(surf, _c("border"), (cx, cy), rad, 2)

        # Episode / step / reward overlay below grid
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
        row("Epsilon:", f"{m.get('epsilon', 1.0):.4f}")
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

        # Progress bar when training
        if self._state in ("training", "paused") and self._thread:
            total = self._thread.params.get("total_timesteps", 1)
            pct   = min(1.0, ts / max(total, 1))
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
        self._btns["SAVE"].draw( surf, self._fb, disabled=self._model is None)
        self._btns["LOAD"].draw( surf, self._fb)
        self._btns["RESET"].draw(surf, self._fb)
        self._btns["EXIT"].draw( surf, self._fb)

        # Status line centred along the bottom edge
        parts = []
        if training and self._thread:
            ts    = self._metrics.get("total_steps", 0)
            total = self._thread.params.get("total_timesteps", 1)
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
    Dashboard().run()


if __name__ == "__main__":
    main()
