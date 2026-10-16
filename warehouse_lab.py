"""What the training lab needs to know about WarehouseEnv, beyond the env itself."""

from lab_params import Param

# WarehouseEnv constructor arguments with a slider.
WAREHOUSE_PARAMS = (
    Param("grid_size", "env", int, 6, 15, 8, label="Grid Size"),
    Param("num_packages", "env", int, 1, 5, 1, label="Packages"),
    Param("max_steps", "env", int, 100, 500, 300, label="Max Steps"),
    Param("step_cost", "env", float, 0.01, 0.5, 0.05, label="Step Cost"),
)
