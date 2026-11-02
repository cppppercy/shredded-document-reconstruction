"""V0.3 protocol: improved learned compatibility plus external-page testing."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.baseline import reconstruct_baseline
from shredded_reconstruction.conditions import CONDITION_NAMES, observation_condition
from shredded_reconstruction.document import generate_document
from shredded_reconstruction.evaluation import order_metrics
from shredded_reconstruction.external import prepare_external_pages
from shredded_reconstruction.learned import document_identity, train_compatibility_model, train_v03_compatibility_model
from shredded_reconstruction.shred import compose_strips, shred_document
from shredded_reconstruction.splits import TRAIN_DOCUMENTS, assert_disjoint_document_splits, document_ids
from shredded_reconstruction.v03_splits import V03_TEST_DOCUMENTS, V03_TRAIN_DOCUMENTS, V03_VALIDATION_DOCUMENTS
from shredded_reconstruction.visualize import comparison_figure, evaluation_grid, shuffled_montage

METHODS = ("traditional_v0_2", "learned_v0_2", "learned_v0_3")
V03_AUGMENTATIONS = tuple(observation_condition(name) for name in CONDITION_NAMES)


def _shuffle_seed(seed: int, condition_index: int) -> int:
    return seed * 19 + condition_index * 7907 + 31


def _method_fields(prefix: str, metrics: dict, seconds: float) -> dict[str, object]:
    return {
        f"{prefix}_adjacency": metrics["adjacency_accuracy"],
        f"{prefix}_position": metrics["position_accuracy"],
        f"{prefix}_exact": metrics["exact_match"],
        f"{prefix}_order": metrics["predicted_original_order"],
        f"{prefix}_seconds": seconds,
    }


def _summary(rows: list[dict[str, object]], method: str) -> dict[str, float | int]:
    if not rows:
        return {}
    denominators = [int(row["strips"]) - 1 for row in rows]
    correct = sum(float(row[f"{method}_adjacency"]) * size for row, size in zip(rows, denominators))
    return {
        "documents": len(rows),
        "exact_recovery_rate": sum(bool(row[f"{method}_exact"]) for row in rows) / len(rows),
        "document_mean_adjacency": sum(float(row[f"{method}_adjacency"]) for row in rows) / len(rows),
        "aggregate_adjacency": correct / sum(denominators),
        "document_mean_position": sum(float(row[f"{method}_position"]) for row in rows) / len(rows),
        "mean_seconds": sum(float(row[f"{method}_seconds"]) for row in rows) / len(rows),
    }


def _grouped(rows: list[dict[str, object]], field: str) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: {method: _summary(group, method) for method in METHODS} for key, group in sorted(grouped.items())}


def _predict(strips, indices, v02_model, v03_model):
    started = perf_counter()
    traditional_order, _ = reconstruct_baseline(strips, max_vertical_shift=3, endpoint_weight=1.0)
    traditional_seconds = perf_counter() - started
    started = perf_counter()
    v02_order, _ = v02_model.reconstruct(strips)
    v02_seconds = perf_counter() - started
    started = perf_counter()
    v03_order, _ = v03_model.reconstruct(strips)
    v03_seconds = perf_counter() - started
    return {
        "traditional_v0_2": (traditional_order, order_metrics(traditional_order, indices), traditional_seconds),
        "learned_v0_2": (v02_order, order_metrics(v02_order, indices), v02_seconds),
        "learned_v0_3": (v03_order, order_metrics(v03_order, indices), v03_seconds),
    }


def _evaluate_synthetic(specs, v02_model, v03_model, *, conditions=CONDITION_NAMES):
    rows, figures = [], []
    for seed, variant, strips in specs:
        for condition_index, name in enumerate(conditions):
            condition = observation_condition(name, seed=seed)
            original = generate_document(seed=seed, variant=variant, with_border=condition.with_border, with_texture=condition.with_texture)
            shredded = shred_document(original, strips, _shuffle_seed(seed, condition_index), condition.disturbance)
            predictions = _predict(shredded.strips, shredded.original_indices, v02_model, v03_model)
            row: dict[str, object] = {
                "family_id": document_identity(seed, variant), "seed": seed, "variant": variant,
                "strips": strips, "condition": name, "shuffle_seed": _shuffle_seed(seed, condition_index),
            }
            for method, (order, metrics, seconds) in predictions.items():
                row.update(_method_fields(method, metrics, seconds))
            rows.append(row)
            figures.append((original, shredded, predictions, row))
            print(json.dumps({key: value for key, value in row.items() if not key.endswith("_order")}))
    return rows, figures


def _evaluate_v03_validation(specs, v03_model, *, conditions=CONDITION_NAMES):
    """Evaluate only the candidate being selected, avoiding redundant controls."""
    rows = []
    for seed, variant, strips in specs:
        for condition_index, name in enumerate(conditions):
            condition = observation_condition(name, seed=seed)
            original = generate_document(seed=seed, variant=variant, with_border=condition.with_border, with_texture=condition.with_texture)
            shredded = shred_document(original, strips, _shuffle_seed(seed, condition_index), condition.disturbance)
            started = perf_counter()
            order, _ = v03_model.reconstruct(shredded.strips)
            seconds = perf_counter() - started
            metrics = order_metrics(order, shredded.original_indices)
            row: dict[str, object] = {
                "family_id": document_identity(seed, variant), "seed": seed, "variant": variant,
                "strips": strips, "condition": name, "shuffle_seed": _shuffle_seed(seed, condition_index),
            }
            row.update(_method_fields("learned_v0_3", metrics, seconds))
            rows.append(row)
    return rows


def _evaluate_external(records, external_dir: Path, v02_model, v03_model):
    rows, figures = [], []
    for index, record in enumerate(records):
        image = Image.open(external_dir / record["rendered_file"]).convert("RGB")
        strips = (8, 10, 12)[index % 3]
        shredded = shred_document(image, strips, 96000 + index)
        predictions = _predict(shredded.strips, shredded.original_indices, v02_model, v03_model)
        row: dict[str, object] = {
            "page_id": record["page_id"], "source_id": record["source_id"], "strips": strips,
            "condition": "external_page_simulated_cut", "shuffle_seed": 96000 + index,
        }
        for method, (order, metrics, seconds) in predictions.items():
            row.update(_method_fields(method, metrics, seconds))
        rows.append(row)
        figures.append((image, shredded, predictions, row))
    return rows, figures


def _save_examples(output_dir: Path, figures):
    success = max(figures, key=lambda item: (item[2]["learned_v0_3"][1]["adjacency_accuracy"], item[2]["learned_v0_3"][1]["exact_match"]))
    failure = min(figures, key=lambda item: (item[2]["learned_v0_3"][1]["adjacency_accuracy"], item[2]["learned_v0_3"][1]["exact_match"]))
    names = []
    images = []
    for label, (original, shredded, predictions, row) in (("success", success), ("failure", failure)):
        baseline_order, baseline_metrics, _ = predictions["traditional_v0_2"]
        learned_order, learned_metrics, _ = predictions["learned_v0_3"]
        figure = comparison_figure(
            original, shuffled_montage(shredded.strips), compose_strips(shredded.strips, baseline_order),
            compose_strips(shredded.strips, learned_order), baseline_metrics, learned_metrics,
        )
        filename = f"{label}-v03-{row['variant']}-{row['strips']}-{row['condition']}-{row['seed']}.png"
        figure.save(output_dir / filename, optimize=True)
        names.append(filename)
        images.append(compose_strips(shredded.strips, learned_order))
    evaluation_grid(images, ["V0.3 best learned case", "V0.3 lowest learned case"], columns=2).save(output_dir / "selected-cases-grid.png", optimize=True)
    return names + ["selected-cases-grid.png"]


def _write_csv(path: Path, rows: list[dict[str, object]]):
    fields = [key for key in rows[0] if not key.endswith("_order")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if not key.endswith("_order")} for row in rows)


def _select_endpoint_multiplier(model):
    candidates = (0.0, 0.8)
    validation_by_multiplier = {}
    for multiplier in candidates:
        rows = _evaluate_v03_validation(V03_VALIDATION_DOCUMENTS, replace(model, endpoint_multiplier=multiplier))
        validation_by_multiplier[str(multiplier)] = _summary(rows, "learned_v0_3")
    selected = max(
        candidates,
        key=lambda value: (
            validation_by_multiplier[str(value)]["aggregate_adjacency"],
            validation_by_multiplier[str(value)]["exact_recovery_rate"],
        ),
    )
    return selected, validation_by_multiplier


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for raw in csv.DictReader(path.open(encoding="utf-8")):
        row: dict[str, object] = dict(raw)
        for key, value in tuple(row.items()):
            if key in {"seed", "strips", "shuffle_seed"}:
                row[key] = int(value)
            elif key.endswith(("_adjacency", "_position", "_seconds")):
                row[key] = float(value)
            elif key.endswith("_exact"):
                row[key] = value == "True"
        rows.append(row)
    return rows


def _save_external_examples(output_dir: Path, examples):
    names = []
    for label, item in (("external-success", max(examples, key=lambda x: x[2]["learned_v0_3"][1]["adjacency_accuracy"])), ("external-failure", min(examples, key=lambda x: x[2]["learned_v0_3"][1]["adjacency_accuracy"]))):
        original, shredded, predictions, row = item
        order, metrics, _ = predictions["learned_v0_3"]
        filename = f"{label}-{row['page_id'].replace(':', '-')}.png"
        comparison_figure(
            original, shuffled_montage(shredded.strips),
            compose_strips(shredded.strips, predictions["traditional_v0_2"][0]),
            compose_strips(shredded.strips, order), predictions["traditional_v0_2"][1], metrics,
        ).save(output_dir / filename, optimize=True)
        names.append(filename)
    return names


def _run_external_only(output_dir: Path, endpoint_multiplier: float) -> int:
    """Resume an interrupted external-page step without rerunning final tests."""
    v02_model = train_compatibility_model(TRAIN_DOCUMENTS)
    v03_model = replace(
        train_v03_compatibility_model(V03_TRAIN_DOCUMENTS, augmentations=V03_AUGMENTATIONS),
        endpoint_multiplier=endpoint_multiplier,
    )
    external_dir = output_dir / "external"
    records = prepare_external_pages(external_dir, ROOT / ".tmp" / "external_sources")
    rows, examples = _evaluate_external(records, external_dir, v02_model, v03_model)
    _write_csv(output_dir / "external_evaluation.csv", rows)
    external_images = _save_external_examples(output_dir, examples)
    synthetic_rows = _read_csv_rows(output_dir / "evaluation.csv")
    selection_path = output_dir / "model_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {"selected_endpoint_multiplier": endpoint_multiplier}
    summary = {
        "protocol": {
            "version": "v0.3", "train_families": len(V03_TRAIN_DOCUMENTS), "validation_families": len(V03_VALIDATION_DOCUMENTS), "final_test_families": len(V03_TEST_DOCUMENTS),
            "conditions": list(CONDITION_NAMES), "split_by": "original document identity (variant:seed)", "splits_disjoint": True,
            "v0_2_model": "clean-only V0.2 training split, row-aligned V0.1 features",
            "v0_3_model": "five named conditions cycled across training families; alignment-aware features; ±3 px non-wrapping alignment",
            "endpoint_multiplier_selection": "chosen only on V0.3 validation families by aggregate adjacency",
            "external_pages": "test-only external full pages, simulated vertical cutting; not physical-shred scans and never training",
        },
        "model_selection_validation": selection,
        "final_test_summary": {method: _summary(synthetic_rows, method) for method in METHODS},
        "by_condition": _grouped(synthetic_rows, "condition"),
        "by_document_type": _grouped(synthetic_rows, "variant"),
        "external_test_summary": {method: _summary(rows, method) for method in METHODS},
        "external_example_images": external_images,
        "cases": synthetic_rows,
        "external_cases": rows,
    }
    (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "analysis.md").write_text(
        "# V0.3 experiment notes\n\n"
        "V0.2 learned compatibility used only clean synthetic pages and did not search vertical offsets. V0.3 cycles all named observations across train families and adds label-free alignment evidence. Endpoint weight is selected on validation only.\n\n"
        "External results use real public-domain PDF pages after simulated vertical cutting. They are not physical shredded-paper scans.\n",
        encoding="utf-8",
    )
    print(json.dumps({"external_test_summary": summary["external_test_summary"], "selected_endpoint_multiplier": endpoint_multiplier}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v0.3"))
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--external-only", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert_disjoint_document_splits(TRAIN_DOCUMENTS, V03_TRAIN_DOCUMENTS, V03_VALIDATION_DOCUMENTS, V03_TEST_DOCUMENTS)

    if args.select_only and args.external_only:
        parser.error("--select-only and --external-only cannot be combined")
    if args.external_only:
        selection = json.loads((args.output_dir / "model_selection.json").read_text(encoding="utf-8"))
        return _run_external_only(args.output_dir, float(selection["selected_endpoint_multiplier"]))

    # Fixed before looking at V0.3 final-test outcomes: V0.2 keeps its original
    # clean-only model; V0.3 sees all five training observations and features
    # a bounded non-wrapping vertical-alignment search.
    v02_model = train_compatibility_model(TRAIN_DOCUMENTS)
    base_v03_model = train_v03_compatibility_model(V03_TRAIN_DOCUMENTS, augmentations=V03_AUGMENTATIONS)
    train_ids = set(base_v03_model.training_document_ids)
    held_out_ids = document_ids(V03_VALIDATION_DOCUMENTS) | document_ids(V03_TEST_DOCUMENTS)
    if train_ids & held_out_ids:
        raise RuntimeError("V0.3 train/held-out document-family overlap")

    # Two qualitatively different settings keep selection affordable while
    # still testing whether synthetic endpoint evidence should be discarded.
    selected_multiplier, validation_by_multiplier = _select_endpoint_multiplier(base_v03_model)
    selection_payload = {"endpoint_multiplier_candidates": validation_by_multiplier, "selected_endpoint_multiplier": selected_multiplier}
    (args.output_dir / "model_selection.json").write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    if args.select_only:
        return 0
    v03_model = replace(base_v03_model, endpoint_multiplier=selected_multiplier)

    test_rows, figures = _evaluate_synthetic(V03_TEST_DOCUMENTS, v02_model, v03_model)
    selected_examples = _save_examples(args.output_dir, figures)
    _write_csv(args.output_dir / "evaluation.csv", test_rows)

    external_rows = []
    external_examples = []
    if not args.skip_external:
        external_dir = args.output_dir / "external"
        records = prepare_external_pages(external_dir, ROOT / ".tmp" / "external_sources")
        external_rows, external_examples = _evaluate_external(records, external_dir, v02_model, v03_model)
        _write_csv(args.output_dir / "external_evaluation.csv", external_rows)
        # Preserve one external success and one failure without manual selection.
        for label, item in (("external-success", max(external_examples, key=lambda x: x[2]["learned_v0_3"][1]["adjacency_accuracy"])), ("external-failure", min(external_examples, key=lambda x: x[2]["learned_v0_3"][1]["adjacency_accuracy"]))):
            original, shredded, predictions, row = item
            order, metrics, _ = predictions["learned_v0_3"]
            filename = f"{label}-{row['page_id'].replace(':', '-')}.png"
            comparison_figure(original, shuffled_montage(shredded.strips), compose_strips(shredded.strips, predictions["traditional_v0_2"][0]), compose_strips(shredded.strips, order), predictions["traditional_v0_2"][1], metrics).save(args.output_dir / filename, optimize=True)

    summary = {
        "protocol": {
            "version": "v0.3", "train_families": len(V03_TRAIN_DOCUMENTS), "validation_families": len(V03_VALIDATION_DOCUMENTS), "final_test_families": len(V03_TEST_DOCUMENTS),
            "conditions": list(CONDITION_NAMES), "split_by": "original document identity (variant:seed)", "splits_disjoint": True,
            "v0_2_model": "clean-only V0.2 training split, row-aligned V0.1 features",
            "v0_3_model": "all five named conditions cycled across training families as augmentation; V0.3 alignment-aware feature set; ±3 px non-wrapping alignment",
            "endpoint_multiplier_selection": "chosen only on V0.3 validation families by aggregate adjacency",
            "training_document_ids": sorted(train_ids),
            "external_pages": "test-only external full pages, simulated vertical cutting; not physical-shred scans and never training",
        },
        "model_selection_validation": selection_payload,
        "final_test_summary": {method: _summary(test_rows, method) for method in METHODS},
        "by_condition": _grouped(test_rows, "condition"),
        "by_document_type": _grouped(test_rows, "variant"),
        "external_test_summary": {method: _summary(external_rows, method) for method in METHODS},
        "selected_example_images": selected_examples,
        "external_example_images": [name for name in (f"external-success-{external_examples[max(range(len(external_examples)), key=lambda i: external_examples[i][2]['learned_v0_3'][1]['adjacency_accuracy'])][3]['page_id'].replace(':', '-')}.png", f"external-failure-{external_examples[min(range(len(external_examples)), key=lambda i: external_examples[i][2]['learned_v0_3'][1]['adjacency_accuracy'])][3]['page_id'].replace(':', '-')}.png") if external_examples],
        "cases": test_rows,
        "external_cases": external_rows,
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "analysis.md").write_text(
        "# V0.3 experiment notes\n\n"
        "The V0.2 learned model was trained only on clean synthetic pages and did not use vertical alignment. V0.3 trains with all declared observation conditions and adds label-free alignment evidence. The endpoint multiplier is selected on validation families only.\n\n"
        "External results are from real public-domain PDF pages after simulated vertical cutting. They are not physical shredded-paper scans.\n",
        encoding="utf-8",
    )
    print(json.dumps({"final_test_summary": summary["final_test_summary"], "external_test_summary": summary["external_test_summary"], "selected_endpoint_multiplier": selected_multiplier}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
