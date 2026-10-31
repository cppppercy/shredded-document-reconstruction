"""Hand-crafted seam compatibility baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from .features import endpoint_costs, pairwise_feature_tensor
from .optimizer import solve_global_order

# Feature weights are intentionally fixed, not tuned on evaluation labels.
BASELINE_WEIGHTS = np.asarray([1.3, 0.7, 2.8, 1.4, 1.0, 0.035, 0.5, 0.8, 0.002])


def baseline_cost_matrix(strips: Sequence[Image.Image], max_vertical_shift: int = 0) -> np.ndarray:
    features = pairwise_feature_tensor(strips, max_vertical_shift=max_vertical_shift)
    costs = np.tensordot(features, BASELINE_WEIGHTS, axes=([2], [0]))
    np.fill_diagonal(costs, np.inf)
    return costs


def reconstruct_baseline(
    strips: Sequence[Image.Image],
    *,
    max_vertical_shift: int = 0,
    endpoint_weight: float = 1.0,
) -> tuple[list[int], float]:
    """Predict an order using only shuffled strip pixels."""
    start, end = endpoint_costs(strips)
    return solve_global_order(baseline_cost_matrix(strips, max_vertical_shift), start * endpoint_weight, end * endpoint_weight)
