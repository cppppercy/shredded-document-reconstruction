"""Prediction glue shared by the UI and its focused tests."""

from __future__ import annotations

from time import perf_counter

from .baseline import reconstruct_baseline
from .evaluation import order_metrics
from .shred import ShreddedDocument


def predict_both_methods(shredded: ShreddedDocument, learned_model) -> dict[str, dict[str, object]]:
    """Run both methods on the exact same shuffled-strip tuple.

    ``original_indices`` is intentionally read only after each method returns,
    when converting an already-predicted order into evaluation metrics.
    """
    started = perf_counter()
    rule_order, _ = reconstruct_baseline(shredded.strips, max_vertical_shift=3, endpoint_weight=1.0)
    rule = {
        "order": rule_order,
        "seconds": perf_counter() - started,
        "metrics": order_metrics(rule_order, shredded.original_indices),
    }
    started = perf_counter()
    learned_order, _ = learned_model.reconstruct(shredded.strips)
    learned = {
        "order": learned_order,
        "seconds": perf_counter() - started,
        "metrics": order_metrics(learned_order, shredded.original_indices),
    }
    return {"rule": rule, "learned": learned}
