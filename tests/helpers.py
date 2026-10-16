"""Helpers shared by the tests: scripted movement, dashboard thread settings."""

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


def thread_params(run_dir, load_path=None, **values):
    """_TrainingThread params from warehouse/agent/run setting values by name."""
    from lab_params import AGENT_PARAMS, RUN_PARAMS, by_target
    from warehouse_lab import WAREHOUSE_PARAMS

    grouped = by_target(WAREHOUSE_PARAMS + AGENT_PARAMS + RUN_PARAMS, values)
    return {**grouped, "load_path": load_path, "run_dir": str(run_dir)}


def deliver(env):
    """Walk to the delivery zone and drop off; return the dropoff step's result."""
    walk_to(env, *env.delivery_zone)
    return env.step(DROPOFF)
