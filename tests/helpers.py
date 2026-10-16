"""Scripted-movement helpers shared by the tests."""

UP, DOWN, LEFT, RIGHT, PICKUP, DROPOFF = range(6)


def walk_to(env, x, y):
    """Step the agent to (x, y) along a Manhattan path; return the step results."""
    results = []
    while env.agent_pos[0] != x:
        results.append(env.step(RIGHT if env.agent_pos[0] < x else LEFT))
    while env.agent_pos[1] != y:
        results.append(env.step(DOWN if env.agent_pos[1] < y else UP))
    return results


def pick_up(env, k):
    """Walk to package k and pick it up; return the pickup step's result."""
    walk_to(env, *env.packages[k][:2])
    return env.step(PICKUP)


def deliver(env):
    """Walk to the delivery zone and drop off; return the dropoff step's result."""
    walk_to(env, *env.delivery_zone)
    return env.step(DROPOFF)
