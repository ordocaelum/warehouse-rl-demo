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
