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
