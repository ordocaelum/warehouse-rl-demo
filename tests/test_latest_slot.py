"""The keep-latest slot between training and display."""

import threading

from latest_slot import LatestSlot
from visualizer_pygame import WarehouseVisualizer


def test_take_returns_the_newest_value_once():
    slot = LatestSlot()
    assert slot.take() == (False, None)
    for i in range(500):
        slot.put(i)
    assert slot.take() == (True, 499)
    assert slot.take() == (False, None)
    assert slot.puts == 500 and slot.overwritten == 499


def test_put_never_blocks_with_a_slow_consumer():
    slot = LatestSlot()
    done = threading.Event()

    def producer():
        for i in range(100_000):
            slot.put(i)
        done.set()

    t = threading.Thread(target=producer)
    t.start()
    t.join(timeout=10)
    assert done.is_set()
    assert slot.take() == (True, 99_999)


def test_visualizer_shows_the_newest_update():
    viz = WarehouseVisualizer()  # not started: no window
    for step in range(1, 1001):
        viz.put_update({"step": step})
    has_update, update = viz._updates.take()
    viz._merge_state(update)
    assert viz._state["step"] == 1000
