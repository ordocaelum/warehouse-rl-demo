"""Renderer for WarehouseEnv's render state (see lab_render.py)."""

from __future__ import annotations

from typing import Optional

import pygame

from lab_render import Renderer

GRID_LINE    = ( 50,  57,  84)
DELIVERY     = ( 50, 135,  58)
DELIVERY_LBL = (120, 215, 120)
AGENT        = (218, 218, 225)
AGENT_CARRY  = (255, 200,  48)
PKG_PENDING  = ( 62, 108, 208)
PKG_DONE     = ( 62, 182,  80)
OUTLINE      = ( 55,  65,  98)


class WarehouseRenderer(Renderer):
    """Square grid, delivery zone, packages and agent, scaled to fit the rect."""

    def __init__(self) -> None:
        self._font: Optional[pygame.font.Font] = None

    def draw(self, surface, rect, state):
        if not state:
            return None
        gs   = int(state["grid_size"])
        cell = max(1, min(rect.w, rect.h) // gs)
        size = cell * gs
        ox   = rect.x + (rect.w - size) // 2
        oy   = rect.y + (rect.h - size) // 2

        # Delivery zone
        dzx, dzy = state["delivery_zone"]
        pygame.draw.rect(surface, DELIVERY, (ox + dzx * cell, oy + dzy * cell, cell, cell))
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 11)
        dl = self._font.render("D", True, DELIVERY_LBL)
        surface.blit(dl, (ox + dzx * cell + (cell - dl.get_width()) // 2,
                          oy + dzy * cell + (cell - dl.get_height()) // 2))

        # Grid lines
        for i in range(gs + 1):
            pygame.draw.line(surface, GRID_LINE, (ox + i * cell, oy), (ox + i * cell, oy + size))
            pygame.draw.line(surface, GRID_LINE, (ox, oy + i * cell), (ox + size, oy + i * cell))

        # Packages
        mg = max(3, cell // 6)
        for px, py, delivered in state["packages"]:
            pygame.draw.rect(
                surface, PKG_DONE if delivered else PKG_PENDING,
                (ox + px * cell + mg, oy + py * cell + mg, cell - 2 * mg, cell - 2 * mg),
                border_radius=max(2, cell // 8),
            )

        # Agent
        ax, ay = state["agent_pos"]
        rad    = max(4, cell // 3)
        centre = (ox + ax * cell + cell // 2, oy + ay * cell + cell // 2)
        pygame.draw.circle(surface, AGENT_CARRY if state["carrying"] >= 0 else AGENT, centre, rad)
        pygame.draw.circle(surface, OUTLINE, centre, rad, 2)
        return pygame.Rect(ox, oy, size, size)

    def status_lines(self, state):
        if not state:
            return []
        carrying  = state["carrying"] >= 0
        delivered = sum(1 for p in state["packages"] if p[2])
        return [
            ("Carrying: YES" if carrying else "Carrying: no", AGENT_CARRY if carrying else None),
            (f"Packages: {delivered}/{len(state['packages'])}", None),
        ]
