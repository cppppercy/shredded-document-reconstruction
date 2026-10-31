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

from .conditions import ObservationCondition
from .document import DOCUMENT_VARIANTS, generate_document
from .features import endpoint_costs, pairwise_feature_tensor
from .optimizer import solve_global_order
from .shred import cut_vertical_strips, disturb_document, disturb_strips


@dataclass
class CompatibilityModel:
    estimator: Pipeline
    training_document_ids: tuple[str, ...]
    sample_count: int
    max_vertical_shift: int = 0
    endpoint_multiplier: float = 1.8

    def cost_matrix(self, strips: Sequence[Image.Image]) -> np.ndarray:
        tensor = pairwise_feature_tensor(strips, max_vertical_shift=self.max_vertical_shift)
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
        return solve_global_order(self.cost_matrix(strips), start * self.endpoint_multiplier, end * self.endpoint_multiplier)


def document_identity(seed: int, variant: str) -> str:
    """Identity of an original document, independent of shredding or nuisance."""
    return f"{variant}:{seed}"


def train_compatibility_model(
    document_specs: Iterable[tuple[int, str, int]] | None = None,
    *,
    augmentations: Iterable[ObservationCondition] | None = None,
    max_vertical_shift: int = 0,
    endpoint_multiplier: float = 1.8,
) -> CompatibilityModel:
    """Train on synthetic documents; callers should keep evaluation seeds disjoint."""
    if document_specs is None:
        document_specs = [
            (1000 + index, DOCUMENT_VARIANTS[index % 3], 8 + index % 5)
            for index in range(24)
        ]
    conditions = tuple(augmentations or (ObservationCondition("clean"),))
    if not conditions:
        raise ValueError("At least one training observation condition is required")
    rows: list[np.ndarray] = []
    labels: list[int] = []
    identifiers: list[str] = []
    for seed, variant, strip_count in document_specs:
        identifiers.append(document_identity(seed, variant))
        for condition in conditions:
            image = generate_document(
                seed=seed,
                variant=variant,
                with_border=condition.with_border,
                with_texture=condition.with_texture,
            )
            # Training labels are defined in source left-to-right order. Apply
            # observation noise but never shuffle here; shuffling is only an
            # inference-time nuisance and would invalidate ``right == left+1``.
            observed = disturb_document(image, condition.disturbance)
            strips = disturb_strips(cut_vertical_strips(observed, strip_count), condition.disturbance)
            tensor = pairwise_feature_tensor(strips, max_vertical_shift=max_vertical_shift)
            for left in range(strip_count):
                for right in range(strip_count):
                    if left != right:
                        rows.append(tensor[left, right])
                        labels.append(int(right == left + 1))
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
    return CompatibilityModel(
        estimator,
        tuple(sorted(set(identifiers))),
        len(labels),
        max_vertical_shift=max_vertical_shift,
        endpoint_multiplier=endpoint_multiplier,
    )
