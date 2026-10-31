"""Run the fixed V0.2 protocol and retain both successful and failed cases."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.conditions import CONDITION_NAMES
from shredded_reconstruction.baseline import reconstruct_baseline
from shredded_reconstruction.evaluation import order_metrics, run_case
from shredded_reconstruction.learned import document_identity, train_compatibility_model
from shredded_reconstruction.shred import compose_strips
from shredded_reconstruction.splits import (
    TEST_DOCUMENTS,
    TRAIN_DOCUMENTS,
    VALIDATION_DOCUMENTS,
    assert_disjoint_document_splits,
    document_ids,
)
from shredded_reconstruction.visualize import comparison_figure, evaluation_grid, shuffled_montage

# Retained only as a clean historical re-check.  V0.1 artifacts are never
# overwritten; their original results remain in artifacts/v0.1/.
V0_1_CASES = (
    (3001, "report", 8, 41), (3002, "table", 9, 42),
    (3003, "mixed", 10, 43), (3004, "sparse", 10, 44),
    (3005, "report", 11, 45), (3006, "table", 12, 46),
    (3007, "mixed", 12, 47), (3008, "sparse", 12, 48),
)


def _shuffle_seed(seed: int, condition_index: int) -> int:
    return seed * 17 + condition_index * 7919 + 13


def _method_fields(prefix: str, metrics: dict, seconds: float) -> dict[str, object]:
    return {
        f"{prefix}_adjacency": metrics["adjacency_accuracy"],
        f"{prefix}_position": metrics["position_accuracy"],
        f"{prefix}_exact": metrics["exact_match"],
        f"{prefix}_order": metrics["predicted_original_order"],
        f"{prefix}_seconds": seconds,
    }


def _summary(rows: list[dict[str, object]], prefix: str) -> dict[str, float | int]:
    if not rows:
        return {}
    denominators = [int(row["strips"]) - 1 for row in rows]
    correct = sum(float(row[f"{prefix}_adjacency"]) * denominator for row, denominator in zip(rows, denominators))
    return {
        "documents": len(rows),
        "exact_recovery_rate": sum(bool(row[f"{prefix}_exact"]) for row in rows) / len(rows),
        "document_mean_adjacency": sum(float(row[f"{prefix}_adjacency"]) for row in rows) / len(rows),
        "aggregate_adjacency": correct / sum(denominators),
        "document_mean_position": sum(float(row[f"{prefix}_position"]) for row in rows) / len(rows),
        "mean_seconds": sum(float(row[f"{prefix}_seconds"]) for row in rows) / len(rows),
    }


def _grouped_summary(rows: list[dict[str, object]], field: str) -> dict[str, dict[str, dict[str, float | int]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    methods = ("baseline_v0_1", "learned_v0_1", "baseline_v0_2", "learned_v0_2")
    return {key: {method: _summary(group, method) for method in methods} for key, group in sorted(groups.items())}


def evaluate_specs(
    specs: tuple[tuple[int, str, int], ...], legacy_model, protocol_model, *, conditions: tuple[str, ...] = CONDITION_NAMES
) -> tuple[list[dict[str, object]], list[tuple[object, object, dict, dict]]]:
    rows: list[dict[str, object]] = []
    figures = []
    for spec in specs:
        seed, variant, strips = spec[:3]
        for condition_index, condition_name in enumerate(conditions):
            shuffle_seed = int(spec[3]) if len(spec) == 4 else _shuffle_seed(seed, condition_index)
            result = run_case(
                seed=seed, variant=variant, strip_count=strips,
                shuffle_seed=shuffle_seed, model=legacy_model,
                condition_name=condition_name,
            )
            started = perf_counter()
            aligned_order, _ = reconstruct_baseline(
                result.shredded.strips, max_vertical_shift=3, endpoint_weight=1.0
            )
            aligned_seconds = perf_counter() - started
            aligned_metrics = order_metrics(aligned_order, result.shredded.original_indices)
            aligned_image = compose_strips(result.shredded.strips, aligned_order)
            started = perf_counter()
            protocol_order, _ = protocol_model.reconstruct(result.shredded.strips)
            protocol_seconds = perf_counter() - started
            protocol_metrics = order_metrics(protocol_order, result.shredded.original_indices)
            row: dict[str, object] = {
                "family_id": document_identity(seed, variant), "seed": seed, "variant": variant,
                "strips": strips, "condition": condition_name,
                "shuffle_seed": shuffle_seed,
            }
            row.update(_method_fields("baseline_v0_1", result.baseline_metrics, result.baseline_seconds))
            row.update(_method_fields("learned_v0_1", result.learned_metrics, result.learned_seconds or 0.0))
            row.update(_method_fields("baseline_v0_2", aligned_metrics, aligned_seconds))
            row.update(_method_fields("learned_v0_2", protocol_metrics, protocol_seconds))
            rows.append(row)
            figures.append((result, aligned_image, aligned_metrics, row))
            print(json.dumps({key: value for key, value in row.items() if not key.endswith("_order")}))
    return rows, figures


def _save_examples(output_dir: Path, figures: list[tuple[object, object, dict, dict]]) -> list[str]:
    """Save the best and the lowest V0.2 score: no manual cherry-picking."""
    success = max(figures, key=lambda item: (item[2]["adjacency_accuracy"], item[2]["exact_match"]))
    failure = min(figures, key=lambda item: (item[2]["adjacency_accuracy"], item[2]["exact_match"]))
    saved: list[str] = []
    for label, (result, improved_image, improved_metrics, row) in (("success", success), ("failure", failure)):
        figure = comparison_figure(
            result.original, shuffled_montage(result.shredded.strips), result.baseline_reconstruction,
            improved_image, result.baseline_metrics, improved_metrics,
        )
        filename = f"{label}-case-{row['variant']}-{row['strips']}-{row['condition']}-{row['seed']}.png"
        figure.save(output_dir / filename, optimize=True)
        saved.append(filename)
    evaluation_grid(
        [success[1], failure[1]],
        [
            f"best V0.2: {success[3]['variant']}/{success[3]['condition']} {success[2]['adjacency_accuracy']:.0%}",
            f"lowest V0.2: {failure[3]['variant']}/{failure[3]['condition']} {failure[2]['adjacency_accuracy']:.0%}",
        ], columns=2,
    ).save(output_dir / "selected-cases-grid.png", optimize=True)
    return saved + ["selected-cases-grid.png"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v0.2"))
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert_disjoint_document_splits(TRAIN_DOCUMENTS, VALIDATION_DOCUMENTS, TEST_DOCUMENTS)

    # V0.1 control: its original clean data and row-aligned features. V0.2
    # learned control uses only the declared training split. The targeted
    # algorithmic improvement is the alignment-aware traditional local score.
    legacy_model = train_compatibility_model()
    protocol_model = train_compatibility_model(TRAIN_DOCUMENTS)
    training_ids = set(protocol_model.training_document_ids)
    if training_ids & document_ids(TEST_DOCUMENTS):
        raise RuntimeError("training/test document-family overlap")

    validation_rows: list[dict[str, object]] = []
    if not args.skip_validation:
        validation_rows, _ = evaluate_specs(VALIDATION_DOCUMENTS, legacy_model, protocol_model)
    test_rows, figures = evaluate_specs(TEST_DOCUMENTS, legacy_model, protocol_model)
    legacy_rows, _ = evaluate_specs(V0_1_CASES, legacy_model, protocol_model, conditions=("clean",))
    examples = _save_examples(args.output_dir, figures)
    methods = ("baseline_v0_1", "learned_v0_1", "baseline_v0_2", "learned_v0_2")
    summary = {
        "protocol": {
            "version": "v0.2", "train_families": len(TRAIN_DOCUMENTS),
            "validation_families": len(VALIDATION_DOCUMENTS), "test_families": len(TEST_DOCUMENTS),
            "test_cases": len(test_rows), "conditions": list(CONDITION_NAMES),
            "split_by": "original document identity (variant:seed)", "splits_disjoint": True,
            "improvement": "thresholded non-wrapping ±3-pixel vertical alignment in the traditional local score",
            "training_document_ids": sorted(training_ids),
        },
        "validation_summary": {method: _summary(validation_rows, method) for method in methods},
        "test_summary": {method: _summary(test_rows, method) for method in methods},
        "by_condition": _grouped_summary(test_rows, "condition"),
        "by_document_type": _grouped_summary(test_rows, "variant"),
        "legacy_eight_clean_recheck": {method: _summary(legacy_rows, method) for method in methods},
        "selected_example_images": examples, "cases": test_rows,
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [key for key in test_rows[0] if not key.endswith("_order")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if not key.endswith("_order")} for row in test_rows)
    (args.output_dir / "validation.jsonl").write_text(
        "" if not validation_rows else "\n".join(json.dumps(row, ensure_ascii=False) for row in validation_rows) + "\n", encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"cases", "protocol"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
