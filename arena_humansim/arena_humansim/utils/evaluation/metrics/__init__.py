from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from arena_humansim.utils.evaluation.metrics.aggregate import (
    calculate_kinematic_metrics,
    calculate_run_collisions,
)
from arena_humansim.utils.evaluation.metrics.pairwise import pairwise_hausdorff


@runtime_checkable
class AggregateMetric(Protocol):
    """Reduces a per-agent or per-trial DataFrame slice to scalar(s)."""

    def __call__(self, df: pd.DataFrame) -> pd.Series | float | int: ...


@runtime_checkable
class PairwiseMetric(Protocol):
    """Symmetric distance between two (n, 2) trajectory point clouds."""

    def __call__(self, traj_a: np.ndarray, traj_b: np.ndarray) -> float: ...


__all__ = [
    "AggregateMetric",
    "PairwiseMetric",
    "calculate_kinematic_metrics",
    "calculate_run_collisions",
    "pairwise_hausdorff",
]
