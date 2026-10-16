"""What the training lab needs to know about WarehouseEnv, beyond the env itself.

Importing this module registers Warehouse-v0 with Gymnasium and its profile
with the lab.
"""

from lab_envs import EnvProfile, register_profile
from lab_params import Param
from warehouse_env import ENV_ID as WAREHOUSE_ID

# WarehouseEnv constructor arguments with a slider.
WAREHOUSE_PARAMS = (
    Param("grid_size", "env", int, 6, 15, 8, label="Grid Size"),
    Param("num_packages", "env", int, 1, 5, 1, label="Packages"),
    Param("max_steps", "env", int, 100, 500, 300, label="Max Steps"),
    Param("step_cost", "env", float, 0.01, 0.5, 0.05, label="Step Cost"),
)


def _renderer():
    from warehouse_render import WarehouseRenderer  # pygame only when drawing

    return WarehouseRenderer()


WAREHOUSE_PROFILE = EnvProfile(
    WAREHOUSE_ID, WAREHOUSE_PARAMS, renderer=_renderer,
    run_name=lambda kw: f"g{kw['grid_size']}_p{kw['num_packages']}",
)
register_profile(WAREHOUSE_PROFILE)
