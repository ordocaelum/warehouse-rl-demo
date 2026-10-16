"""Algorithm-specific training metrics, keyed by name.

Each SB3 algorithm records its own metrics in its logger under its own
names: DQN ``rollout/exploration_rate`` and ``train/loss``; PPO
``train/entropy_loss``, ``train/approx_kl`` and so on. The training
callbacks forward whatever is there instead of assuming a particular
algorithm, so no metric (``epsilon`` included) is required to exist.
"""

from __future__ import annotations

import numbers
from typing import Dict, List, Tuple


def collect_algorithm_metrics(logger, into: Dict[str, float]) -> Dict[str, float]:
    """Merge the numeric values ``logger`` holds into ``into`` and return it.

    The logger forgets its values each time it writes them out, so ``into``
    keeps each metric's latest value between writes.
    """
    for name, value in logger.name_to_value.items():
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            into[name] = float(value)
    return into


def display_metrics(metrics: Dict[str, float], limit: int) -> List[Tuple[str, str]]:
    """Up to ``limit`` metrics as ``(short name, formatted value)``, in recording order."""
    return [(name.rsplit("/", 1)[-1], f"{value:.4g}")
            for name, value in list(metrics.items())[:limit]]
