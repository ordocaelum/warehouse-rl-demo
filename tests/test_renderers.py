"""Renderers draw an env's render state into a rect; the displays delegate to them."""

import pytest

pygame = pytest.importorskip("pygame")

import warehouse_render as wr  # noqa: E402

BLACK = (0, 0, 0)
STATE = {"grid_size": 4, "agent_pos": [0, 0], "carrying": -1,
         "packages": [[2, 1, 0]], "delivery_zone": [3, 3]}


@pytest.fixture(autouse=True)
def _pygame(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    yield
    pygame.quit()


def _surface(w=400, h=400):
    surf = pygame.Surface((w, h))
    surf.fill(BLACK)
    return surf


def _rgb(surf, x, y):
    return tuple(surf.get_at((x, y)))[:3]


def test_warehouse_renderer_draws_each_entity_in_its_cell():
    surf = _surface()
    drawn = wr.WarehouseRenderer().draw(surf, pygame.Rect(0, 0, 400, 400), STATE)
    assert drawn == pygame.Rect(0, 0, 400, 400)  # 100 px cells
    assert _rgb(surf, 50, 50) == wr.AGENT
    assert _rgb(surf, 250, 150) == wr.PKG_PENDING
    assert _rgb(surf, 305, 305) == wr.DELIVERY


def test_carried_and_delivered_colours():
    surf = _surface()
    state = dict(STATE, agent_pos=[2, 1], carrying=0, packages=[[2, 1, 0], [1, 3, 1]])
    wr.WarehouseRenderer().draw(surf, pygame.Rect(0, 0, 400, 400), state)
    assert _rgb(surf, 250, 150) == wr.AGENT_CARRY
    assert _rgb(surf, 150, 350) == wr.PKG_DONE


def test_drawing_is_centred_inside_the_rect_and_stays_in_it():
    surf = _surface(400, 300)
    drawn = wr.WarehouseRenderer().draw(surf, pygame.Rect(10, 20, 300, 200), STATE)
    assert drawn == pygame.Rect(60, 20, 200, 200)  # 50 px cells, centred horizontally
    # + 1 px for the closing grid line on the right and bottom edges
    inked = pygame.Rect(drawn.x, drawn.y, drawn.w + 1, drawn.h + 1)
    for x in range(surf.get_width()):
        for y in range(surf.get_height()):
            if not inked.collidepoint(x, y):
                assert _rgb(surf, x, y) == BLACK, (x, y)


def test_nothing_is_drawn_before_the_first_update():
    surf = _surface(50, 50)
    assert wr.WarehouseRenderer().draw(surf, pygame.Rect(0, 0, 50, 50), None) is None
    assert all(_rgb(surf, x, y) == BLACK for x in range(50) for y in range(50))
    assert wr.WarehouseRenderer().status_lines(None) == []


def test_status_lines():
    state = dict(STATE, carrying=0, packages=[[0, 0, 0], [3, 3, 1]])
    assert wr.WarehouseRenderer().status_lines(state) == [
        ("Carrying: YES", wr.AGENT_CARRY), ("Packages: 1/2", None)]


def test_dashboard_draws_the_forwarded_state_with_its_renderer():
    from ui_dashboard import CENTER_W, LEFT_W, Dashboard

    calls = []

    class Recording(wr.WarehouseRenderer):
        def draw(self, surface, rect, state):
            calls.append((rect, state))
            return super().draw(surface, rect, state)

    d = Dashboard(renderer=Recording())
    d._draw_center()  # before the first update
    d._metrics = {"render_state": STATE}
    d._draw_center()
    (_, before), (rect, state) = calls
    assert before is None
    assert state is STATE
    assert LEFT_W <= rect.left and rect.right <= LEFT_W + CENTER_W


def test_observation_renderer_draws_one_signed_bar_per_component():
    from lab_render import ObservationRenderer

    r = ObservationRenderer()
    surf = _surface(400, 300)
    state = {"observation": [1.0, -2.0, 0.0], "low": [-2.0, None, None], "high": [2.0, None, None]}
    drawn = r.draw(surf, pygame.Rect(0, 0, 400, 300), state)
    assert drawn.h == 3 * 48 and drawn.w == 400
    colours = [{_rgb(surf, x, drawn.y + i * 48 + 24) for x in range(70, 310)} for i in range(3)]
    assert r.POS in colours[0] and r.NEG not in colours[0]  # +1 in [-2, 2]: half right
    assert r.NEG in colours[1] and r.POS not in colours[1]  # unbounded: scaled to itself
    assert r.POS not in colours[2] and r.NEG not in colours[2]
    assert r.draw(_surface(), pygame.Rect(0, 0, 10, 10), None) is None
