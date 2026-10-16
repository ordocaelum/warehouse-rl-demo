"""
PyGame-based live visualizer for warehouse RL training.

The window runs on the main thread (pygame/SDL requires this on macOS) via
run(); training runs in a background thread and pushes updates via
put_update(). Updates go through a keep-latest slot (see latest_slot.py), so
the display always shows the newest state.
"""

import collections
import threading
import time
from typing import List, Optional

from latest_slot import LatestSlot

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CELL_SIZE = 60          # pixels per grid cell (base zoom level)
PANEL_WIDTH = 280       # right-hand metrics panel
FPS_CAP = 60
GRID_SIZE = 8
# Completed-episode returns kept for the sparkline.
REWARD_HISTORY_LEN = 200

# Colours (R, G, B)
COL_BG           = (30,  30,  40)
COL_GRID_LINE    = (60,  60,  80)
COL_DELIVERY     = (80, 160,  80)
COL_AGENT_EMPTY  = (220, 220, 220)
COL_AGENT_CARRY  = (255, 200,  50)
COL_PKG_PENDING  = ( 80, 130, 220)
COL_PKG_DONE     = ( 80, 200, 100)
COL_TEXT         = (200, 210, 230)
COL_TEXT_POS     = ( 80, 220,  80)
COL_TEXT_NEG     = (220,  80,  80)
COL_PANEL_BG     = ( 20,  20,  30)
COL_HEADER       = (160, 180, 240)
COL_PAUSED_BG    = (200, 120,  30)
COL_PAUSED_TEXT  = (255, 255, 255)
COL_BORDER       = ( 80,  90, 120)


class WarehouseVisualizer:
    """
    PyGame visualizer for the warehouse RL environment.

    Typical usage from a training script::

        viz = WarehouseVisualizer(grid_size=8)
        # start training in a background thread, then on the main thread:
        viz.run()                    # blocks until the window closes or stop()

        # inside the training callback (training thread):
        viz.put_update({
            "episode":      episode_num,
            "step":         step_num,
            "reward":       step_reward,
            "ep_reward":    cumulative_episode_reward,
            "avg_reward":   moving_average_reward,       # optional
            "epsilon":      exploration_rate,            # optional
            "agent_pos":    [x, y],
            "packages":     [[x, y, delivered], ...],
            "carrying":     -1 or package_index,
            "grid_size":    8,
            "delivery_zone": [x, y],
        })

        # once per finished episode:
        viz.put_episode_reward(episode_return)

        viz.stop()                   # from any thread: close the window
    """

    def __init__(self, grid_size: int = GRID_SIZE):
        if not PYGAME_AVAILABLE:
            raise RuntimeError(
                "pygame is required for the live visualizer. "
                "Install it with:  pip install pygame"
            )

        self.grid_size = grid_size
        # Newest per-step update; training never waits on the display.
        self._updates = LatestSlot()
        self._stop_event = threading.Event()
        self._running = False  # True while run() has the window open

        # State mirrored from the training loop
        self._state = {
            "episode":   0,
            "step":      0,
            "reward":    0.0,
            "ep_reward": 0.0,
            "avg_reward": None,
            "epsilon":   1.0,
            "agent_pos": [0, 0],
            "packages":  [],
            "carrying":  -1,
            "grid_size": grid_size,
            "delivery_zone": None,  # None until the first update: bottom-right
        }

        # Visualizer-local mutable properties
        self._paused = False
        self._step_requested = False
        self._zoom = 1.0       # zoom multiplier
        self._pan_x = 0        # pixel pan offsets
        self._pan_y = 0
        self._dragging = False
        self._drag_start = (0, 0)
        self._pan_start = (0, 0)
        # Completed-episode returns. Written by the training thread, read by
        # the render thread, hence the lock. Never dropped, unlike updates.
        self._reward_history = collections.deque(maxlen=REWARD_HISTORY_LEN)
        self._reward_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Ask run() to close the window and return. Safe from any thread."""
        self._stop_event.set()

    def put_update(self, state: dict) -> None:
        """
        Non-blocking push of a training-state snapshot to the visualizer.
        Replaces any update not yet drawn, so each frame shows the newest state.
        """
        self._updates.put(state)

    def put_episode_reward(self, episode_reward: float) -> None:
        """Record the return of a finished episode for the reward sparkline."""
        with self._reward_lock:
            self._reward_history.append(float(episode_reward))

    def reward_history(self) -> List[float]:
        """Returns of the most recent finished episodes, oldest first."""
        with self._reward_lock:
            return list(self._reward_history)

    def is_paused(self) -> bool:
        """Return True while training is paused by the user."""
        return self._paused

    def consume_step_request(self) -> bool:
        """
        Return True (and clear the flag) if the user pressed S to step
        one episode while paused.
        """
        if self._step_requested:
            self._step_requested = False
            return True
        return False

    def is_running(self) -> bool:
        """Return True if the visualizer window is still open."""
        return self._running

    # ------------------------------------------------------------------
    # Rendering loop (call on the main thread)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the window and render until it is closed or stop() is called.

        Call this on the main thread: SDL's window and event handling must
        run there on macOS. Training should run in another thread.
        """
        self._running = True
        pygame.init()
        cell = int(CELL_SIZE * self._zoom)
        grid_px = cell * self.grid_size
        width  = grid_px + PANEL_WIDTH
        height = grid_px
        screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Warehouse RL – Live Visualizer")
        clock  = pygame.time.Clock()

        font_large  = pygame.font.SysFont("monospace", 18, bold=True)
        font_medium = pygame.font.SysFont("monospace", 15)
        font_small  = pygame.font.SysFont("monospace", 12)

        try:
            while not self._stop_event.is_set():
                # --- newest update, if any ---------------------------------
                has_update, update = self._updates.take()
                if has_update:
                    self._merge_state(update)

                # --- handle pygame events -----------------------------------
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._stop_event.set()

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self._paused = not self._paused
                        elif event.key == pygame.K_s and self._paused:
                            self._step_requested = True
                        elif event.key == pygame.K_r:
                            self._zoom  = 1.0
                            self._pan_x = 0
                            self._pan_y = 0
                        elif event.key == pygame.K_UP:
                            self._zoom = min(3.0, self._zoom + 0.15)
                        elif event.key == pygame.K_DOWN:
                            self._zoom = max(0.3, self._zoom - 0.15)

                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._dragging   = True
                        self._drag_start = event.pos
                        self._pan_start  = (self._pan_x, self._pan_y)

                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        self._dragging = False

                    elif event.type == pygame.MOUSEMOTION and self._dragging:
                        dx = event.pos[0] - self._drag_start[0]
                        dy = event.pos[1] - self._drag_start[1]
                        self._pan_x = self._pan_start[0] + dx
                        self._pan_y = self._pan_start[1] + dy

                    elif event.type == pygame.MOUSEWHEEL:
                        self._zoom = max(0.3, min(3.0, self._zoom + event.y * 0.1))

                    elif event.type == pygame.VIDEORESIZE:
                        screen = pygame.display.set_mode(
                            (event.w, event.h), pygame.RESIZABLE
                        )

                # --- draw ---------------------------------------------------
                cell = int(CELL_SIZE * self._zoom)
                grid_px = cell * self.grid_size
                w, h = screen.get_size()

                self._draw_background(screen, w, h)
                self._draw_grid(screen, cell, grid_px)
                self._draw_panel(screen, font_large, font_medium, font_small,
                                 grid_px, w, h)
                if self._paused:
                    self._draw_paused_banner(screen, font_large, w)

                pygame.display.flip()
                clock.tick(FPS_CAP)

        finally:
            self._running = False
            pygame.quit()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _merge_state(self, update: dict) -> None:
        """Merge an incoming training-state snapshot into local state."""
        self._state.update(update)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_background(self, screen: "pygame.Surface", w: int, h: int) -> None:
        """Adaptive background colour based on recent reward trend."""
        avg = self._state.get("avg_reward")
        if avg is None:
            base = COL_BG
        else:
            # Clamp avg reward to [-20, +20] and map to brightness.
            # This range covers the typical reward scale for the default warehouse env
            # (delivery reward +10, step penalty -0.05, max episode ~300 steps).
            t = max(0.0, min(1.0, (avg + 20) / 40))
            r = int(COL_BG[0] * (1 - t) + 50 * t)
            g = int(COL_BG[1] * (1 - t) + 60 * t)
            b = int(COL_BG[2] * (1 - t) + 70 * t)
            base = (r, g, b)
        screen.fill(base)

    def _draw_grid(self, screen: "pygame.Surface", cell: int, grid_px: int) -> None:
        """Render the grid, delivery zone, packages, and agent."""
        state = self._state
        packages  = state.get("packages",  [])
        agent_pos = state.get("agent_pos", [0, 0])
        carrying  = state.get("carrying",  -1)
        gs        = state.get("grid_size", self.grid_size)

        ox = self._pan_x  # origin offsets
        oy = self._pan_y

        # --- delivery zone --------------------------------------------------
        dz_x, dz_y = state.get("delivery_zone") or (gs - 1, gs - 1)
        pygame.draw.rect(
            screen, COL_DELIVERY,
            (ox + dz_x * cell, oy + dz_y * cell, cell, cell)
        )

        # --- grid lines -----------------------------------------------------
        for col in range(gs + 1):
            x = ox + col * cell
            pygame.draw.line(screen, COL_GRID_LINE, (x, oy), (x, oy + gs * cell))
        for row in range(gs + 1):
            y = oy + row * cell
            pygame.draw.line(screen, COL_GRID_LINE, (ox, y), (ox + gs * cell, y))

        # --- packages -------------------------------------------------------
        for pkg in packages:
            px, py, delivered = int(pkg[0]), int(pkg[1]), bool(pkg[2])
            colour = COL_PKG_DONE if delivered else COL_PKG_PENDING
            margin = max(4, cell // 6)
            pygame.draw.rect(
                screen, colour,
                (ox + px * cell + margin,
                 oy + py * cell + margin,
                 cell - 2 * margin,
                 cell - 2 * margin),
                border_radius=max(2, cell // 8)
            )

        # --- agent ----------------------------------------------------------
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        colour = COL_AGENT_CARRY if carrying >= 0 else COL_AGENT_EMPTY
        radius = max(4, cell // 3)
        cx_ = ox + ax * cell + cell // 2
        cy_ = oy + ay * cell + cell // 2
        pygame.draw.circle(screen, colour, (cx_, cy_), radius)
        # thin border
        pygame.draw.circle(screen, COL_BORDER, (cx_, cy_), radius, 2)

    def _draw_panel(
        self,
        screen: "pygame.Surface",
        font_large: "pygame.font.Font",
        font_medium: "pygame.font.Font",
        font_small: "pygame.font.Font",
        grid_px: int,
        w: int,
        h: int,
    ) -> None:
        """Render the right-hand metrics panel."""
        state = self._state
        panel_x = grid_px + self._pan_x
        # Panel background
        pygame.draw.rect(
            screen, COL_PANEL_BG,
            (panel_x, 0, w - panel_x, h)
        )
        pygame.draw.line(
            screen, COL_BORDER,
            (panel_x, 0), (panel_x, h), 2
        )

        x = panel_x + 12
        y = 16

        def label(text: str, colour=COL_HEADER, font=font_large) -> None:
            nonlocal y
            surf = font.render(text, True, colour)
            screen.blit(surf, (x, y))
            y += surf.get_height() + 4

        def value(key: str, fmt: str, colour=COL_TEXT, font=font_medium) -> None:
            nonlocal y
            v = state.get(key)
            if v is None:
                text = f"{key}: N/A"
            else:
                text = fmt.format(v)
            surf = font.render(text, True, colour)
            screen.blit(surf, (x, y))
            y += surf.get_height() + 4

        label("== METRICS ==")
        y += 4

        value("episode",   "Episode:  {:>6d}")
        value("step",      "Step:     {:>6d}")

        ep_r = state.get("ep_reward", 0.0)
        r_colour = COL_TEXT_POS if ep_r >= 0 else COL_TEXT_NEG
        surf = font_medium.render(f"Ep Reward:{ep_r:>+8.2f}", True, r_colour)
        screen.blit(surf, (x, y))
        y += surf.get_height() + 4

        avg = state.get("avg_reward")
        if avg is not None:
            a_colour = COL_TEXT_POS if avg >= 0 else COL_TEXT_NEG
            surf = font_medium.render(f"Avg(50ep):{avg:>+8.2f}", True, a_colour)
            screen.blit(surf, (x, y))
            y += surf.get_height() + 4

        eps = state.get("epsilon", 1.0)
        surf = font_medium.render(f"Epsilon:  {eps:>8.4f}", True, COL_TEXT)
        screen.blit(surf, (x, y))
        y += surf.get_height() + 4

        # steps whose state was replaced before a frame could draw it
        surf = font_small.render(
            f"Not drawn: {self._updates.overwritten:,} steps", True, COL_BORDER
        )
        screen.blit(surf, (x, y))
        y += surf.get_height() + 4

        # carrying status
        y += 8
        label("== AGENT ==", font=font_medium)
        carrying = state.get("carrying", -1)
        c_text   = "Carrying: YES" if carrying >= 0 else "Carrying: no"
        c_colour = COL_AGENT_CARRY if carrying >= 0 else COL_TEXT
        surf = font_medium.render(c_text, True, c_colour)
        screen.blit(surf, (x, y))
        y += surf.get_height() + 4

        pkgs = state.get("packages", [])
        delivered = sum(1 for p in pkgs if p[2])
        surf = font_medium.render(
            f"Packages: {delivered}/{len(pkgs)}", True, COL_TEXT
        )
        screen.blit(surf, (x, y))
        y += surf.get_height() + 4

        # mini reward history sparkline
        history = self.reward_history()
        if len(history) >= 2:
            y += 12
            label("== REWARD HISTORY ==", font=font_small)
            self._draw_sparkline(screen, font_small, history, x, y, PANEL_WIDTH - 24, 60)
            y += 72

        # controls legend
        y = h - 150
        label("== CONTROLS ==", font=font_small, colour=COL_BORDER)
        for line in [
            "SPACE  pause / resume",
            "S      step episode (paused)",
            "UP/DN  zoom in / out",
            "WHEEL  zoom",
            "DRAG   pan",
            "R      reset view",
        ]:
            surf = font_small.render(line, True, COL_BORDER)
            screen.blit(surf, (x, y))
            y += surf.get_height() + 2

    def _draw_sparkline(
        self,
        screen: "pygame.Surface",
        font: "pygame.font.Font",
        history: List[float],
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> None:
        """Draw a tiny sparkline of recent episode rewards."""
        data = history[-w:]  # up to w most recent episode returns
        if len(data) < 2:
            return
        mn = min(data)
        mx = max(data)
        rng = mx - mn if mx != mn else 1.0

        points = []
        for i, v in enumerate(data):
            px_ = x + int(i / (len(data) - 1) * w)
            py_ = y + h - int((v - mn) / rng * h)
            points.append((px_, py_))

        if len(points) >= 2:
            pygame.draw.lines(screen, COL_HEADER, False, points, 1)

        # axis labels
        mn_surf = font.render(f"{mn:.1f}", True, COL_TEXT_NEG)
        mx_surf = font.render(f"{mx:.1f}", True, COL_TEXT_POS)
        screen.blit(mn_surf, (x, y + h + 2))
        screen.blit(mx_surf, (x + w - mx_surf.get_width(), y + h + 2))

    def _draw_paused_banner(
        self,
        screen: "pygame.Surface",
        font: "pygame.font.Font",
        w: int,
    ) -> None:
        """Semi-transparent PAUSED banner across the top of the grid."""
        banner_h = 36
        banner   = pygame.Surface((w, banner_h), pygame.SRCALPHA)
        banner.fill((*COL_PAUSED_BG, 200))
        screen.blit(banner, (0, 0))
        text = "  [PAUSED]  SPACE to resume  |  S to step one episode  "
        surf = font.render(text, True, COL_PAUSED_TEXT)
        screen.blit(surf, ((w - surf.get_width()) // 2, (banner_h - surf.get_height()) // 2))
