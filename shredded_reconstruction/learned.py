"""A lightweight learned compatibility model built with scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable, Sequence

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .document import DOCUMENT_VARIANTS, generate_document
from .features import endpoint_costs, pairwise_feature_tensor
from .optimizer import solve_global_order
from .shred import cut_vertical_strips


@dataclass
class CompatibilityModel:
    estimator: Pipeline
    training_document_ids: tuple[str, ...]
    sample_count: int

    def cost_matrix(self, strips: Sequence[Image.Image]) -> np.ndarray:
        tensor = pairwise_feature_tensor(strips)
        count = len(strips)
        costs = np.full((count, count), np.inf, dtype=np.float64)
        pairs = [(left, right) for left in range(count) for right in range(count) if left != right]
        feature_rows = np.stack([tensor[left, right] for left, right in pairs])
        probabilities = self.estimator.predict_proba(feature_rows)[:, 1]
        for (left, right), probability in zip(pairs, probabilities, strict=True):
            costs[left, right] = -math.log(float(np.clip(probability, 1e-6, 1.0)))
        return costs

    def reconstruct(self, strips: Sequence[Image.Image]) -> tuple[list[int], float]:
        start, end = endpoint_costs(strips)
        # Learned pair costs are on a larger logarithmic scale.
        return solve_global_order(self.cost_matrix(strips), start * 1.8, end * 1.8)


def train_compatibility_model(
    document_specs: Iterable[tuple[int, str, int]] | None = None,
) -> CompatibilityModel:
    """Train on synthetic documents; callers should keep evaluation seeds disjoint."""
    if document_specs is None:
        document_specs = [
            (1000 + index, DOCUMENT_VARIANTS[index % 3], 8 + index % 5)
            for index in range(24)
        ]
    rows: list[np.ndarray] = []
    labels: list[int] = []
    identifiers: list[str] = []
    for seed, variant, strip_count in document_specs:
        image = generate_document(seed=seed, variant=variant)
        strips = cut_vertical_strips(image, strip_count)
        tensor = pairwise_feature_tensor(strips)
        for left in range(strip_count):
            for right in range(strip_count):
                if left == right:
                    continue
                rows.append(tensor[left, right])
                labels.append(int(right == left + 1))
        identifiers.append(f"{variant}:{seed}:{strip_count}")
    x = np.stack(rows)
    y = np.asarray(labels)
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=0,
                    C=0.7,
                ),
            ),
        ]
    )
    estimator.fit(x, y)
    return CompatibilityModel(estimator, tuple(identifiers), len(labels))
