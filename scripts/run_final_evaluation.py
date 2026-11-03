"""Run the pre-specified two-method final audit on unseen document families."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.baseline import reconstruct_baseline
from shredded_reconstruction.conditions import CONDITION_NAMES, observation_condition
from shredded_reconstruction.document import generate_document
from shredded_reconstruction.evaluation import order_metrics
from shredded_reconstruction.learned import train_v03_compatibility_model
from shredded_reconstruction.research_analysis import comparison_table, family_bootstrap_interval
from shredded_reconstruction.shred import shred_document
from shredded_reconstruction.splits import assert_disjoint_document_splits
from shredded_reconstruction.v03_splits import V03_TEST_DOCUMENTS, V03_TRAIN_DOCUMENTS, V03_VALIDATION_DOCUMENTS
from shredded_reconstruction.v04_splits import V04_CONFIRMATION_DOCUMENTS, V04_TRAIN_DOCUMENTS, V04_VALIDATION_DOCUMENTS
from shredded_reconstruction.v05_splits import FINAL_AUDIT_DOCUMENTS


def _shuffle_seed(seed: int, condition_index: int) -> int:
    return seed * 29 + condition_index * 911 + 59


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "final")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert_disjoint_document_splits(
        V03_TRAIN_DOCUMENTS, V03_VALIDATION_DOCUMENTS, V03_TEST_DOCUMENTS,
        V04_TRAIN_DOCUMENTS, V04_VALIDATION_DOCUMENTS, V04_CONFIRMATION_DOCUMENTS,
        FINAL_AUDIT_DOCUMENTS,
    )
    model = train_v03_compatibility_model(
        V04_TRAIN_DOCUMENTS,
        augmentations=tuple(observation_condition(name) for name in CONDITION_NAMES),
        endpoint_multiplier=0.0,
    )
    rows: list[dict[str, object]] = []
    for seed, variant, strip_count in FINAL_AUDIT_DOCUMENTS:
        for condition_index, condition_name in enumerate(CONDITION_NAMES):
            condition = observation_condition(condition_name, seed=seed)
            original = generate_document(seed=seed, variant=variant, with_border=condition.with_border, with_texture=condition.with_texture)
            shredded = shred_document(original, strip_count, _shuffle_seed(seed, condition_index), condition.disturbance)
            started = perf_counter()
            rule_order, _ = reconstruct_baseline(shredded.strips, max_vertical_shift=3, endpoint_weight=1.0)
            rule_seconds = perf_counter() - started
            started = perf_counter()
            learned_order, _ = model.reconstruct(shredded.strips)
            learned_seconds = perf_counter() - started
            rule = order_metrics(rule_order, shredded.original_indices)
            learned = order_metrics(learned_order, shredded.original_indices)
            rows.append({
                "family_id": f"{variant}:{seed}", "seed": seed, "variant": variant, "strips": strip_count,
                "condition": condition_name, "shuffle_seed": _shuffle_seed(seed, condition_index),
                "traditional_v0_2_adjacency": rule["adjacency_accuracy"], "traditional_v0_2_exact": rule["exact_match"],
                "traditional_v0_2_order": rule["predicted_original_order"], "traditional_v0_2_seconds": rule_seconds,
                "learned_v03_full_adjacency": learned["adjacency_accuracy"], "learned_v03_full_exact": learned["exact_match"],
                "learned_v03_full_order": learned["predicted_original_order"], "learned_v03_full_seconds": learned_seconds,
            })
    fields = [field for field in rows[0] if not field.endswith("_order")]
    with (args.output_dir / "independent_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    output = {
        "protocol": {
            "purpose": "final untouched audit; methods and parameters were fixed before this test",
            "families": len(FINAL_AUDIT_DOCUMENTS), "cases": len(rows), "conditions": list(CONDITION_NAMES),
            "training": "V0.4 training families only; no final-audit page is used in training or selection",
            "split_by": "original document identity (variant:seed)", "splits_disjoint": True,
        },
        "overall": comparison_table(rows),
        "by_document_type": comparison_table(rows, "variant"),
        "by_condition": comparison_table(rows, "condition"),
        "by_type_and_condition": comparison_table(rows, "variant", "condition"),
        "family_bootstrap_95": family_bootstrap_interval(rows),
        "cases": rows,
    }
    (args.output_dir / "independent_evaluation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
