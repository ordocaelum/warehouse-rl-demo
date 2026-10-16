"""Stand-ins for the pygame visualizer, for driving the training callbacks."""


class RecordingVisualizer:
    """Implements the visualizer interface the callback uses; records updates."""

    def __init__(self):
        self.updates = []

    def put_update(self, state):
        self.updates.append(state)

    def is_paused(self):
        return False

    def is_running(self):
        return True

    def consume_step_request(self):
        return False


class SteppingVisualizer(RecordingVisualizer):
    """Permanently paused; presses S ``presses`` times, then closes the window.

    Records (episode, step) of the frame on screen each time training blocks.
    """

    def __init__(self, presses):
        super().__init__()
        self.presses = presses
        self.blocked_at = []

    def is_paused(self):
        return True

    def is_running(self):
        return self.presses > 0

    def consume_step_request(self):
        last = self.updates[-1]
        self.blocked_at.append((last["episode"], last["step"]))
        self.presses -= 1
        return True
