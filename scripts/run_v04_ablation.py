"""V0.4 ablations on new document families, plus new external-page checks.

The configurations are fixed before the confirmation set is run.  The V0.4
validation rows are descriptive checks only: no parameter is selected from
either validation or confirmation results in this script.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.baseline import reconstruct_baseline
from shredded_reconstruction.conditions import CONDITION_NAMES, observation_condition
from shredded_reconstruction.document import _font, generate_document
from shredded_reconstruction.evaluation import order_metrics
from shredded_reconstruction.external import V04_EXTERNAL_SOURCES, prepare_pages_for_sources
from shredded_reconstruction.learned import document_identity, train_compatibility_model, train_v03_compatibility_model
from shredded_reconstruction.shred import compose_strips, shred_document
from shredded_reconstruction.splits import TRAIN_DOCUMENTS, TEST_DOCUMENTS, VALIDATION_DOCUMENTS, assert_disjoint_document_splits
from shredded_reconstruction.v03_splits import V03_TEST_DOCUMENTS, V03_TRAIN_DOCUMENTS, V03_VALIDATION_DOCUMENTS
from shredded_reconstruction.v04_splits import V04_CONFIRMATION_DOCUMENTS, V04_TRAIN_DOCUMENTS, V04_VALIDATION_DOCUMENTS
from shredded_reconstruction.visualize import comparison_figure, shuffled_montage


METHODS = (
    "traditional_v0_2",
    "traditional_no_endpoint",
    "learned_v03_no_augmentation",
    "learned_v03_without_alignment",
    "learned_v03_full",
)
V03_AUGMENTATIONS = tuple(observation_condition(name) for name in CONDITION_NAMES)


def _shuffle_seed(seed: int, condition_index: int) -> int:
    return seed * 23 + condition_index * 8111 + 47


def _method_fields(prefix: str, metrics: dict[str, object], seconds: float) -> dict[str, object]:
    return {
        f"{prefix}_adjacency": metrics["adjacency_accuracy"],
        f"{prefix}_position": metrics["position_accuracy"],
        f"{prefix}_exact": metrics["exact_match"],
        f"{prefix}_order": metrics["predicted_original_order"],
        f"{prefix}_seconds": seconds,
    }


def _summary(rows: list[dict[str, object]], method: str) -> dict[str, float | int]:
    denominators = [int(row["strips"]) - 1 for row in rows]
    correct = sum(float(row[f"{method}_adjacency"]) * size for row, size in zip(rows, denominators, strict=True))
    return {
        "documents": len(rows),
        "exact_recovery_rate": sum(bool(row[f"{method}_exact"]) for row in rows) / len(rows),
        "document_mean_adjacency": sum(float(row[f"{method}_adjacency"]) for row in rows) / len(rows),
        "aggregate_adjacency": correct / sum(denominators),
        "document_mean_position": sum(float(row[f"{method}_position"]) for row in rows) / len(rows),
        "mean_seconds": sum(float(row[f"{method}_seconds"]) for row in rows) / len(rows),
    }


def _grouped(rows: list[dict[str, object]], field: str) -> dict[str, dict[str, dict[str, float | int]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: {method: _summary(group, method) for method in METHODS} for name, group in sorted(groups.items())}


def _train_models():
    """Train fixed ablation variants on V0.4 training families only."""
    return {
        "learned_v03_no_augmentation": train_compatibility_model(
            V04_TRAIN_DOCUMENTS,
            augmentations=(observation_condition("clean"),),
            max_vertical_shift=3,
            endpoint_multiplier=0.0,
            feature_version="v0.3",
            cycle_augmentations=True,
        ),
        "learned_v03_without_alignment": train_compatibility_model(
            V04_TRAIN_DOCUMENTS,
            augmentations=V03_AUGMENTATIONS,
            max_vertical_shift=0,
            endpoint_multiplier=0.0,
            feature_version="v0.1",
            cycle_augmentations=True,
        ),
        "learned_v03_full": train_v03_compatibility_model(
            V04_TRAIN_DOCUMENTS,
            augmentations=V03_AUGMENTATIONS,
            endpoint_multiplier=0.0,
        ),
    }


def _predict(strips, indices, models):
    predictions: dict[str, tuple[list[int], dict[str, object], float]] = {}
    for name, max_shift, endpoint in (
        ("traditional_v0_2", 3, 1.0),
        ("traditional_no_endpoint", 3, 0.0),
    ):
        started = perf_counter()
        order, _ = reconstruct_baseline(strips, max_vertical_shift=max_shift, endpoint_weight=endpoint)
        predictions[name] = (order, order_metrics(order, indices), perf_counter() - started)
    for name, model in models.items():
        started = perf_counter()
        order, _ = model.reconstruct(strips)
        predictions[name] = (order, order_metrics(order, indices), perf_counter() - started)
    return predictions


def _evaluate_synthetic(specs, models):
    rows, figures = [], []
    for seed, variant, strip_count in specs:
        for condition_index, condition_name in enumerate(CONDITION_NAMES):
            condition = observation_condition(condition_name, seed=seed)
            original = generate_document(seed=seed, variant=variant, with_border=condition.with_border, with_texture=condition.with_texture)
            shredded = shred_document(original, strip_count, _shuffle_seed(seed, condition_index), condition.disturbance)
            predictions = _predict(shredded.strips, shredded.original_indices, models)
            row: dict[str, object] = {
                "family_id": document_identity(seed, variant), "seed": seed, "variant": variant,
                "strips": strip_count, "condition": condition_name,
                "shuffle_seed": _shuffle_seed(seed, condition_index),
            }
            for method, (order, metrics, seconds) in predictions.items():
                row.update(_method_fields(method, metrics, seconds))
            rows.append(row)
            figures.append((original, shredded, predictions, row))
            print(json.dumps({key: value for key, value in row.items() if not key.endswith("_order")}))
    return rows, figures


def _evaluate_external(records, external_dir: Path, models):
    rows, figures = [], []
    for index, record in enumerate(records):
        image = Image.open(external_dir / record["rendered_file"]).convert("RGB")
        strip_count = (8, 10, 12)[index % 3]
        shredded = shred_document(image, strip_count, 120000 + index)
        predictions = _predict(shredded.strips, shredded.original_indices, models)
        row: dict[str, object] = {
            "page_id": record["page_id"], "source_id": record["source_id"], "strips": strip_count,
            "condition": "external_page_simulated_cut", "shuffle_seed": 120000 + index,
        }
        for method, (order, metrics, seconds) in predictions.items():
            row.update(_method_fields(method, metrics, seconds))
        rows.append(row)
        figures.append((image, shredded, predictions, row))
    return rows, figures


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [key for key in rows[0] if not key.endswith("_order")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if not key.endswith("_order")})


def _bar_chart(path: Path, summary: dict[str, dict[str, float | int]], title: str) -> None:
    labels = list(summary)
    image = Image.new("RGB", (1240, 610), "white")
    draw = ImageDraw.Draw(image)
    draw.text((48, 36), title, fill=(24, 42, 62), font=_font(28, bold=True))
    draw.text((48, 77), "Metric: document-mean adjacent-pair recovery; all bars use the same independent confirmation set.", fill=(74, 88, 105), font=_font(16))
    left, top, right, bottom = 110, 150, 1180, 510
    draw.line((left, top, left, bottom), fill=(80, 96, 114), width=2)
    draw.line((left, bottom, right, bottom), fill=(80, 96, 114), width=2)
    for tick in range(0, 101, 20):
        y = bottom - (bottom - top) * tick / 100
        draw.line((left, y, right, y), fill=(224, 229, 235), width=1)
        draw.text((48, y - 9), f"{tick}%", fill=(74, 88, 105), font=_font(14))
    colors = [(69, 125, 181), (127, 150, 176), (214, 140, 65), (170, 104, 161), (57, 142, 105)]
    width = (right - left) / len(labels)
    for index, label in enumerate(labels):
        value = float(summary[label]["document_mean_adjacency"])
        x0 = int(left + index * width + width * 0.22)
        x1 = int(left + (index + 1) * width - width * 0.22)
        y0 = int(bottom - (bottom - top) * value)
        draw.rectangle((x0, y0, x1, bottom), fill=colors[index % len(colors)])
        draw.text((x0, y0 - 29), f"{value:.1%}", fill=(24, 42, 62), font=_font(18, bold=True))
        words = label.replace("traditional_", "trad. ").replace("learned_v03_", "ML ").replace("_", " ")
        draw.multiline_text((int(left + index * width + 8), 528), words, fill=(47, 63, 82), font=_font(14), align="center", spacing=2)
    image.save(path, optimize=True)


def _save_examples(output_dir: Path, figures) -> list[str]:
    full = "learned_v03_full"
    success = max(figures, key=lambda item: (item[2][full][1]["adjacency_accuracy"], item[2][full][1]["exact_match"]))
    failure = min(figures, key=lambda item: (item[2][full][1]["adjacency_accuracy"], item[2][full][1]["exact_match"]))
    result = []
    for label, (original, shredded, predictions, row) in (("success", success), ("failure", failure)):
        traditional_order, traditional_metrics, _ = predictions["traditional_v0_2"]
        learned_order, learned_metrics, _ = predictions[full]
        figure = comparison_figure(
            original, shuffled_montage(shredded.strips), compose_strips(shredded.strips, traditional_order),
            compose_strips(shredded.strips, learned_order), traditional_metrics, learned_metrics,
        )
        identity = str(row.get("seed", row.get("page_id", "page"))).replace(":", "-")
        filename = f"{label}-v04-{row.get('variant', row.get('source_id'))}-{row['strips']}-{row['condition']}-{identity}.png"
        figure.save(output_dir / filename, optimize=True)
        result.append(filename)
    return result


def _write_analysis(path: Path, validation, confirmation, external, examples: list[str]) -> None:
    full = confirmation["learned_v03_full"]
    no_aug = confirmation["learned_v03_no_augmentation"]
    no_align = confirmation["learned_v03_without_alignment"]
    endpoint = confirmation["traditional_v0_2"]
    no_endpoint = confirmation["traditional_no_endpoint"]
    text = f"""# V0.4 independent ablation and confirmation

## Protocol

- Training families: 24 new V0.4 families (seeds 9100+); validation families: 12 (10100+); confirmation families: 24 (11100+).
- Every family is evaluated under five fixed observations (`clean`, `no_assists`, `scan`, `edge_loss`, `vertical_shift`), yielding 120 confirmation cases.
- The five configurations were fixed before confirmation. The validation table is retained as a descriptive check; no parameter was selected from confirmation results.
- `traditional_no_endpoint` removes only the traditional endpoint auxiliary cost. `learned_v03_no_augmentation` sees clean observations only. `learned_v03_without_alignment` retains nuisance augmentation but uses V0.1 features and zero vertical search. `learned_v03_full` uses the V0.3 feature/alignment/augmentation configuration with endpoint multiplier fixed at 0.0.

## Confirmation results

| Method | Document-mean adjacency | Exact recovery |
| --- | ---: | ---: |
| Traditional V0.2 | {endpoint['document_mean_adjacency']:.2%} | {endpoint['exact_recovery_rate']:.2%} |
| Traditional without endpoint cost | {no_endpoint['document_mean_adjacency']:.2%} | {no_endpoint['exact_recovery_rate']:.2%} |
| V0.3 learned, no nuisance augmentation | {no_aug['document_mean_adjacency']:.2%} | {no_aug['exact_recovery_rate']:.2%} |
| V0.3 learned, no alignment feature/search | {no_align['document_mean_adjacency']:.2%} | {no_align['exact_recovery_rate']:.2%} |
| V0.3 learned, full | {full['document_mean_adjacency']:.2%} | {full['exact_recovery_rate']:.2%} |

The full-model differences relative to no-augmentation and no-alignment variants are respectively {(full['document_mean_adjacency'] - no_aug['document_mean_adjacency']):+.2%} and {(full['document_mean_adjacency'] - no_align['document_mean_adjacency']):+.2%}. These are confirmation-set observations, not tuning decisions.

## External pages

New sources are full public pages followed by simulated vertical cuts, not physical-shred scans. Six pages from three new sources were held outside every model split. The complete metrics and provenance are in `external_evaluation.csv` and `external/external_sources.json`.

## Saved reconstruction cases

{chr(10).join(f'- `{name}`' for name in examples)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "v0.4")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--external-only", action="store_true", help="Add external-page results to an existing V0.4 confirmation run.")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    assert_disjoint_document_splits(
        TRAIN_DOCUMENTS, VALIDATION_DOCUMENTS, TEST_DOCUMENTS,
        V03_TRAIN_DOCUMENTS, V03_VALIDATION_DOCUMENTS, V03_TEST_DOCUMENTS,
        V04_TRAIN_DOCUMENTS, V04_VALIDATION_DOCUMENTS, V04_CONFIRMATION_DOCUMENTS,
    )
    models = _train_models()
    if args.external_only:
        prior_path = output_dir / "ablation.json"
        if not prior_path.exists():
            raise FileNotFoundError("--external-only requires an existing ablation.json confirmation run")
        external_dir = output_dir / "external"
        records = prepare_pages_for_sources(external_dir, ROOT / ".tmp" / "v04_external_sources", V04_EXTERNAL_SOURCES)
        external_rows, external_figures = _evaluate_external(records, external_dir, models)
        _write_csv(output_dir / "external_evaluation.csv", external_rows)
        external_names = _save_examples(output_dir, external_figures)
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        prior["external_summary"] = {method: _summary(external_rows, method) for method in METHODS}
        prior["external_cases"] = external_rows
        prior["examples"] = [
            name for name in prior.get("examples", []) if "external_page_simulated_cut" not in name
        ] + external_names
        prior_path.write_text(json.dumps(prior, indent=2), encoding="utf-8")
        _write_analysis(
            output_dir / "analysis.md", prior["validation_summary"], prior["confirmation_summary"],
            prior["external_summary"], prior["examples"],
        )
        print(json.dumps({"external": prior["external_summary"]}, indent=2))
        return 0
    validation_rows, _ = _evaluate_synthetic(V04_VALIDATION_DOCUMENTS, models)
    confirmation_rows, confirmation_figures = _evaluate_synthetic(V04_CONFIRMATION_DOCUMENTS, models)
    _write_csv(output_dir / "validation.csv", validation_rows)
    _write_csv(output_dir / "confirmation.csv", confirmation_rows)

    validation = {method: _summary(validation_rows, method) for method in METHODS}
    confirmation = {method: _summary(confirmation_rows, method) for method in METHODS}
    _bar_chart(output_dir / "ablation-confirmation.png", confirmation, "V0.4 independent confirmation: ablation comparison")
    example_names = _save_examples(output_dir, confirmation_figures)

    external_rows: list[dict[str, object]] = []
    external_example_names: list[str] = []
    if not args.skip_external:
        external_dir = output_dir / "external"
        records = prepare_pages_for_sources(external_dir, ROOT / ".tmp" / "v04_external_sources", V04_EXTERNAL_SOURCES)
        external_rows, external_figures = _evaluate_external(records, external_dir, models)
        _write_csv(output_dir / "external_evaluation.csv", external_rows)
        external_example_names = _save_examples(output_dir, external_figures)

    summary = {
        "protocol": {
            "version": "v0.4", "train_families": len(V04_TRAIN_DOCUMENTS),
            "validation_families": len(V04_VALIDATION_DOCUMENTS), "confirmation_families": len(V04_CONFIRMATION_DOCUMENTS),
            "confirmation_cases": len(confirmation_rows), "conditions": list(CONDITION_NAMES),
            "split_by": "original document identity (variant:seed)", "splits_disjoint": True,
            "selection_rule": "All ablation configurations and endpoint multiplier 0.0 were fixed before the V0.4 confirmation set; no confirmation result was used for selection.",
            "external_pages": "New public full-page sources are test-only simulated vertical cuts; they are not physical shredded-paper scans.",
        },
        "methods": {
            "traditional_v0_2": "V0.2 hand-crafted compatibility, non-wrapping +/-3 px alignment, endpoint cost weight 1.0.",
            "traditional_no_endpoint": "Same traditional method with only endpoint cost removed.",
            "learned_v03_no_augmentation": "V0.3 12-D alignment-aware features/search, but clean observations only during training.",
            "learned_v03_without_alignment": "Nuisance augmentation retained; V0.1 9-D features and no vertical-search alignment.",
            "learned_v03_full": "V0.3 five-condition cycling, 12-D alignment-aware features/search, endpoint multiplier 0.0.",
        },
        "validation_summary": validation,
        "confirmation_summary": confirmation,
        "confirmation_by_condition": _grouped(confirmation_rows, "condition"),
        "confirmation_by_document_type": _grouped(confirmation_rows, "variant"),
        "external_summary": {method: _summary(external_rows, method) for method in METHODS} if external_rows else {},
        "examples": example_names + external_example_names,
        "validation_cases": validation_rows,
        "confirmation_cases": confirmation_rows,
        "external_cases": external_rows,
    }
    (output_dir / "ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_analysis(output_dir / "analysis.md", validation, confirmation, summary["external_summary"], example_names + external_example_names)
    print(json.dumps({"confirmation": confirmation, "external": summary["external_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
