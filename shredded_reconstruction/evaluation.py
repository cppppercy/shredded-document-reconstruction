"""End-to-end reconstruction and honest order metrics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from time import perf_counter

from PIL import Image

from .baseline import reconstruct_baseline
from .conditions import observation_condition
from .document import generate_document
from .learned import CompatibilityModel
from .shred import ShreddedDocument, compose_strips, disturb_document, shred_document


def order_metrics(predicted_positions: Sequence[int], original_indices: Sequence[int]) -> dict[str, float | bool | list[int]]:
    """Evaluate a prediction after reconstruction; labels never enter prediction."""
    count = len(original_indices)
    if sorted(predicted_positions) != list(range(count)):
        raise ValueError("predicted_positions must permute shuffled input positions")
    predicted_original = [int(original_indices[position]) for position in predicted_positions]
    correct = list(range(count))
    position_accuracy = sum(a == b for a, b in zip(predicted_original, correct, strict=True)) / count
    correct_adjacencies = sum(
        right == left + 1 for left, right in zip(predicted_original, predicted_original[1:])
    )
    adjacency_accuracy = correct_adjacencies / max(1, count - 1)
    concordant = sum(
        predicted_original[i] < predicted_original[j]
        for i in range(count)
        for j in range(i + 1, count)
    )
    relative_order_accuracy = concordant / max(1, count * (count - 1) // 2)
    return {
        "predicted_original_order": predicted_original,
        "correct_original_order": correct,
        "position_accuracy": float(position_accuracy),
        "adjacency_accuracy": float(adjacency_accuracy),
        "relative_order_accuracy": float(relative_order_accuracy),
        "exact_match": predicted_original == correct,
    }


@dataclass
class CaseResult:
    seed: int
    variant: str
    strip_count: int
    original: Image.Image
    shredded: ShreddedDocument
    baseline_order: list[int]
    baseline_reconstruction: Image.Image
    baseline_metrics: dict[str, float | bool | list[int]]
    learned_order: list[int] | None
    learned_reconstruction: Image.Image | None
    learned_metrics: dict[str, float | bool | list[int]] | None
    condition: str = "clean"
    observed_document: Image.Image | None = None
    baseline_seconds: float = 0.0
    learned_seconds: float | None = None


def run_case(
    *,
    seed: int,
    variant: str,
    strip_count: int,
    shuffle_seed: int,
    model: CompatibilityModel | None = None,
    condition_name: str = "clean",
    baseline_max_vertical_shift: int = 0,
    baseline_endpoint_weight: float = 1.0,
) -> CaseResult:
    condition = observation_condition(condition_name, seed=seed)
    original = generate_document(
        seed=seed,
        variant=variant,
        with_border=condition.with_border,
        with_texture=condition.with_texture,
    )
    observed = disturb_document(original, condition.disturbance)
    shredded = shred_document(
        original,
        strip_count=strip_count,
        shuffle_seed=shuffle_seed,
        disturbance=condition.disturbance,
    )
    started = perf_counter()
    baseline_order, _ = reconstruct_baseline(
        shredded.strips,
        max_vertical_shift=baseline_max_vertical_shift,
        endpoint_weight=baseline_endpoint_weight,
    )
    baseline_seconds = perf_counter() - started
    baseline_image = compose_strips(shredded.strips, baseline_order)
    baseline_metrics = order_metrics(baseline_order, shredded.original_indices)
    if model is None:
        learned_order = None
        learned_image = None
        learned_metrics = None
    else:
        started = perf_counter()
        learned_order, _ = model.reconstruct(shredded.strips)
        learned_seconds = perf_counter() - started
        learned_image = compose_strips(shredded.strips, learned_order)
        learned_metrics = order_metrics(learned_order, shredded.original_indices)
    if model is None:
        learned_seconds = None
    return CaseResult(
        seed,
        variant,
        strip_count,
        original,
        shredded,
        baseline_order,
        baseline_image,
        baseline_metrics,
        learned_order,
        learned_image,
        learned_metrics,
        condition_name,
        observed,
        baseline_seconds,
        learned_seconds,
    )


def run_image_case(
    image: Image.Image,
    *,
    strip_count: int,
    shuffle_seed: int,
    model: CompatibilityModel | None = None,
    condition_name: str = "clean",
    baseline_max_vertical_shift: int = 0,
    baseline_endpoint_weight: float = 1.0,
) -> CaseResult:
    """Run the same pipeline on an uploaded/external full document image.

    The input image is cut and shuffled before either solver runs. The true
    order stored by ``ShreddedDocument`` is used strictly after prediction for
    the displayed metrics, exactly as it is for synthetic cases.
    """
    condition = observation_condition(condition_name, seed=shuffle_seed)
    original = image.convert("RGB")
    shredded = shred_document(
        original,
        strip_count=strip_count,
        shuffle_seed=shuffle_seed,
        disturbance=condition.disturbance,
    )
    started = perf_counter()
    baseline_order, _ = reconstruct_baseline(
        shredded.strips,
        max_vertical_shift=baseline_max_vertical_shift,
        endpoint_weight=baseline_endpoint_weight,
    )
    baseline_seconds = perf_counter() - started
    baseline_image = compose_strips(shredded.strips, baseline_order)
    baseline_metrics = order_metrics(baseline_order, shredded.original_indices)
    learned_order: list[int] | None = None
    learned_image: Image.Image | None = None
    learned_metrics: dict[str, float | bool | list[int]] | None = None
    learned_seconds: float | None = None
    if model is not None:
        started = perf_counter()
        learned_order, _ = model.reconstruct(shredded.strips)
        learned_seconds = perf_counter() - started
        learned_image = compose_strips(shredded.strips, learned_order)
        learned_metrics = order_metrics(learned_order, shredded.original_indices)
    return CaseResult(
        -1,
        "uploaded",
        strip_count,
        original,
        shredded,
        baseline_order,
        baseline_image,
        baseline_metrics,
        learned_order,
        learned_image,
        learned_metrics,
        condition_name,
        original,
        baseline_seconds,
        learned_seconds,
    )
