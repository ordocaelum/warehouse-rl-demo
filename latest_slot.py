"""A one-value mailbox between a fast producer and a slower consumer.

The training thread produces one state update per env step, often thousands
per second; the display consumes at most one per frame (60 per second). A
bounded queue that rejects new items when full shows a stale state once it
fills up. This slot keeps only the newest value instead: the producer never
blocks or waits, and the consumer always gets the current state.
"""

import threading
from typing import Any, Tuple


class LatestSlot:
    """Holds the most recent value put into it until it is taken."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Any = None
        self._full = False
        self.puts = 0          # values put in so far
        self.overwritten = 0   # values replaced before anyone took them

    def put(self, value: Any) -> None:
        """Store ``value``, replacing any value not yet taken. Never blocks."""
        with self._lock:
            if self._full:
                self.overwritten += 1
            self._value = value
            self._full = True
            self.puts += 1

    def take(self) -> Tuple[bool, Any]:
        """Return ``(True, value)`` if a value arrived since the last take, else ``(False, None)``."""
        with self._lock:
            if not self._full:
                return False, None
            value, self._value, self._full = self._value, None, False
            return True, value
