"""How the training lab draws an environment.

The dashboard and the visualizer never draw an environment themselves: they
hand its render state (the ``info["render_state"]`` the env pushed, forwarded
unchanged by the training callback) to a renderer, one per environment.

A renderer subclasses :class:`Renderer` and implements ``draw``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

Colour = Tuple[int, int, int]


class Renderer:
    """Draws one environment's render state."""

    def draw(self, surface, rect, state: Optional[dict]):
        """Draw ``state`` inside ``rect`` (a ``pygame.Rect``) of ``surface``.

        ``state`` is None before the first update. Returns the
        ``pygame.Rect`` actually drawn in, or None if nothing was drawn.
        """
        raise NotImplementedError

    def status_lines(self, state: Optional[dict]) -> List[Tuple[str, Optional[Colour]]]:
        """Short ``(text, colour or None)`` lines describing ``state`` for a side panel."""
        return []


class ObservationRenderer(Renderer):
    """Any env's observation, one horizontal bar per component.

    Draws the state lab_envs.ObservationRenderState reports for envs with
    no render state of their own. A bar starts at the middle of the
    component's range; a component with no finite bounds is scaled to the
    largest magnitude it has shown so far.
    """

    MAX_BARS = 32
    POS   = ( 90, 150, 232)
    NEG   = (220, 110,  70)
    AXIS  = ( 55,  65,  98)
    LABEL = (150, 160, 185)

    def __init__(self) -> None:
        self._peak: dict = {}  # component index -> largest |value| seen
        self._font = None

    def _fraction(self, i: int, v: float, lo, hi) -> float:
        if lo is not None and hi is not None and hi > lo:
            f = (v - (lo + hi) / 2) / ((hi - lo) / 2)
        else:
            peak = self._peak[i] = max(self._peak.get(i, 0.0), abs(v))
            f = v / peak if peak else 0.0
        return max(-1.0, min(1.0, f))

    def draw(self, surface, rect, state):
        import pygame

        if not state or not state.get("observation"):
            return None
        obs  = state["observation"][: self.MAX_BARS]
        n    = len(obs)
        low  = state.get("low") or [None] * n
        high = state.get("high") or [None] * n
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 12)

        row_h  = max(8, min(48, rect.h // n))
        top    = rect.y + (rect.h - row_h * n) // 2
        bar_x  = rect.x + 70           # room for the labels on the left
        bar_w  = rect.w - 70 - 90      # and the values on the right
        mid    = bar_x + bar_w // 2
        for i, v in enumerate(obs):
            y = top + i * row_h
            f = self._fraction(i, v, low[i], high[i])
            w = int(abs(f) * bar_w / 2)
            x = mid if f >= 0 else mid - w
            pygame.draw.rect(surface, self.POS if f >= 0 else self.NEG,
                             (x, y + row_h // 5, w, row_h - 2 * (row_h // 5)))
            name = self._font.render(f"obs[{i}]", True, self.LABEL)
            surface.blit(name, (rect.x, y + (row_h - name.get_height()) // 2))
            val = self._font.render(f"{v:+.4g}", True, self.LABEL)
            surface.blit(val, (rect.right - val.get_width(), y + (row_h - val.get_height()) // 2))
        pygame.draw.line(surface, self.AXIS, (mid, top), (mid, top + row_h * n))
        return pygame.Rect(rect.x, top, rect.w, row_h * n)
