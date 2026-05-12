"""
PyGame-based live visualizer for warehouse RL training.

Runs in a separate thread and receives state updates via a thread-safe queue.
The training loop runs on the main thread (CPU) and pushes updates via put_update().
"""

import queue
import threading
import time
from typing import List, Optional

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
# Buffer enough frames for ~2 seconds at 60 FPS to absorb bursts without
# slowing the training thread; excess updates are silently dropped.
UPDATE_QUEUE_SIZE = 120

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
        viz.start()                  # opens window in background thread

        # inside training callback:
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
        })

        viz.stop()                   # signal visualizer to close
    """

    def __init__(self, grid_size: int = GRID_SIZE):
        if not PYGAME_AVAILABLE:
            raise RuntimeError(
                "pygame is required for the live visualizer. "
                "Install it with:  pip install pygame"
            )

        self.grid_size = grid_size
        self._queue: queue.Queue = queue.Queue(maxsize=UPDATE_QUEUE_SIZE)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

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
        self._reward_history: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the visualizer in a background daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the visualizer to shut down and wait for its thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def put_update(self, state: dict) -> None:
        """
        Non-blocking push of a training-state snapshot to the visualizer.
        Drops the update if the queue is full to prevent training slowdown.
        """
        try:
            self._queue.put_nowait(state)
        except queue.Full:
            pass  # visualizer is behind; skip frame

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
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal rendering loop (runs in background thread)
    # ------------------------------------------------------------------

    def _run(self) -> None:
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
                # --- drain the update queue ---------------------------------
                try:
                    while True:
                        update = self._queue.get_nowait()
                        self._merge_state(update)
                except queue.Empty:
                    pass

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
            pygame.quit()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _merge_state(self, update: dict) -> None:
        """Merge an incoming training-state snapshot into local state."""
        self._state.update(update)
        ep_r = update.get("ep_reward")
        if ep_r is not None:
            self._reward_history.append(ep_r)
            if len(self._reward_history) > 200:
                self._reward_history = self._reward_history[-200:]

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
        dz_x = gs - 1
        dz_y = gs - 1
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
        if len(self._reward_history) >= 2:
            y += 12
            label("== REWARD HISTORY ==", font=font_small)
            self._draw_sparkline(screen, font_small, x, y, PANEL_WIDTH - 24, 60)
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
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> None:
        """Draw a tiny sparkline of recent episode rewards."""
        data = self._reward_history[-w:]  # up to w most recent reward data points
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
