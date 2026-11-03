"""Run a retrospective, controlled weight comparison on the V0.4 validation set.

This is a diagnostic experiment only.  It does not select parameters and never
reads the frozen final audit.  The first two methods use exactly the same
nine seam descriptors, zero endpoint term, zero vertical alignment, and the
same shuffled input; only fixed manual weights versus learned logistic weights
differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.baseline import reconstruct_baseline
from shredded_reconstruction.conditions import CONDITION_NAMES, observation_condition
from shredded_reconstruction.document import generate_document
from shredded_reconstruction.evaluation import order_metrics
from shredded_reconstruction.learned import train_compatibility_model, train_v03_compatibility_model
from shredded_reconstruction.research_analysis import comparison_table
from shredded_reconstruction.shred import shred_document
from shredded_reconstruction.splits import assert_disjoint_document_splits
from shredded_reconstruction.v04_splits import V04_CONFIRMATION_DOCUMENTS, V04_TRAIN_DOCUMENTS, V04_VALIDATION_DOCUMENTS
from shredded_reconstruction.v05_splits import FINAL_AUDIT_DOCUMENTS


METHODS = (
    "manual_9f_no_alignment_no_endpoint",
    "learned_9f_no_alignment_no_endpoint",
    "manual_9f_aligned_no_endpoint",
    "learned_12f_aligned_no_endpoint",
)


def _shuffle_seed(seed: int, condition_index: int) -> int:
    # Separate from historical V0.4 and final permutations. This is a fixed
    # validation-only diagnostic, declared before execution.
    return seed * 31 + condition_index * 1223 + 71


def _summary(rows: list[dict[str, object]], method: str) -> dict[str, float | int]:
    return {
        "documents": len(rows),
        "document_mean_adjacency": sum(float(row[f"{method}_adjacency"]) for row in rows) / len(rows),
        "exact_recovery_rate": sum(bool(row[f"{method}_exact"]) for row in rows) / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "final" / "development_controls.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert_disjoint_document_splits(V04_TRAIN_DOCUMENTS, V04_VALIDATION_DOCUMENTS, V04_CONFIRMATION_DOCUMENTS, FINAL_AUDIT_DOCUMENTS)
    augmentations = tuple(observation_condition(name) for name in CONDITION_NAMES)
    learned_9f = train_compatibility_model(
        V04_TRAIN_DOCUMENTS,
        augmentations=augmentations,
        max_vertical_shift=0,
        endpoint_multiplier=0.0,
        feature_version="v0.1",
        cycle_augmentations=True,
    )
    learned_12f = train_v03_compatibility_model(
        V04_TRAIN_DOCUMENTS,
        augmentations=augmentations,
        endpoint_multiplier=0.0,
    )
    rows: list[dict[str, object]] = []
    for seed, variant, strips in V04_VALIDATION_DOCUMENTS:
        for condition_index, condition_name in enumerate(CONDITION_NAMES):
            condition = observation_condition(condition_name, seed=seed)
            page = generate_document(seed=seed, variant=variant, with_border=condition.with_border, with_texture=condition.with_texture)
            shredded = shred_document(page, strips, _shuffle_seed(seed, condition_index), condition.disturbance)
            predictions = {
                METHODS[0]: reconstruct_baseline(shredded.strips, max_vertical_shift=0, endpoint_weight=0.0)[0],
                METHODS[1]: learned_9f.reconstruct(shredded.strips)[0],
                METHODS[2]: reconstruct_baseline(shredded.strips, max_vertical_shift=3, endpoint_weight=0.0)[0],
                METHODS[3]: learned_12f.reconstruct(shredded.strips)[0],
            }
            row: dict[str, object] = {
                "family_id": f"{variant}:{seed}", "seed": seed, "variant": variant,
                "strips": strips, "condition": condition_name, "shuffle_seed": _shuffle_seed(seed, condition_index),
            }
            for method, order in predictions.items():
                metrics = order_metrics(order, shredded.original_indices)
                row[f"{method}_adjacency"] = metrics["adjacency_accuracy"]
                row[f"{method}_exact"] = metrics["exact_match"]
            rows.append(row)
    output = {
        "protocol": {
            "purpose": "retrospective V0.4 validation diagnostic; not used for model selection and not a final-test result",
            "families": len(V04_VALIDATION_DOCUMENTS), "cases": len(rows), "conditions": list(CONDITION_NAMES),
            "training": "V0.4 training families only", "evaluation": "V0.4 validation families only",
            "final_audit_read": False, "same_shuffled_input_per_row": True,
            "scikit_learn": sklearn.__version__,
        },
        "methods": {
            METHODS[0]: "fixed manual nine-feature weights; zero endpoint term; zero vertical alignment",
            METHODS[1]: "logistic learned weights on the same nine features; same augmentation, zero endpoint and zero alignment",
            METHODS[2]: "fixed manual nine-feature weights; no endpoint term; +/-3px alignment",
            METHODS[3]: "logistic learned weights on 12 alignment-aware features; same augmentation; no endpoint term; +/-3px alignment",
        },
        "summary": {method: _summary(rows, method) for method in METHODS},
        "by_condition": {
            method: [
                {"condition": name, "samples": sum(row["condition"] == name for row in rows),
                 "adjacency": sum(float(row[f"{method}_adjacency"]) for row in rows if row["condition"] == name) / sum(row["condition"] == name for row in rows)}
                for name in CONDITION_NAMES
            ]
            for method in METHODS
        },
        "cases": rows,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
