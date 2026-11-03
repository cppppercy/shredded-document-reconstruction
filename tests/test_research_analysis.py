from pathlib import Path
import json

import joblib

from shredded_reconstruction.research_analysis import (
    case_categories,
    comparison_table,
    family_bootstrap_interval,
    load_confirmation,
    validate_final_case_manifest,
)
from shredded_reconstruction.model_artifact import TRAINING_SPLIT, validate_model_metadata
from shredded_reconstruction.v04_splits import V04_TRAIN_DOCUMENTS


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_dashboard_uses_all_saved_rows_and_real_group_counts() -> None:
    rows = load_confirmation(ROOT / "artifacts" / "v0.4" / "ablation.json")
    assert len(rows) == 120
    assert sum(len(group) for group in case_categories(rows).values()) == len(rows)
    by_type = comparison_table(rows, "variant")
    by_condition = comparison_table(rows, "condition")
    by_combination = comparison_table(rows, "variant", "condition")
    assert len(by_type) == 4 and all(row["samples"] == 30 for row in by_type)
    assert len(by_condition) == 5 and all(row["samples"] == 24 for row in by_condition)
    assert len(by_combination) == 20 and all(row["samples"] == 6 for row in by_combination)


def test_final_audit_rows_are_loadable_without_mixing_v04_results() -> None:
    rows = load_confirmation(ROOT / "artifacts" / "final" / "independent_evaluation.json")
    assert len(rows) == 120
    assert {row["seed"] for row in rows}.isdisjoint({11100, 11101})


def test_interactive_model_metadata_matches_the_serialized_model() -> None:
    directory = ROOT / "artifacts" / "final" / "interactive_model"
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(directory / "compatibility_model.joblib")
    assert metadata["feature_version"] == model.feature_version == "v0.3"
    assert metadata["max_vertical_shift"] == model.max_vertical_shift == 3
    assert metadata["training_document_ids"] == list(model.training_document_ids)
    assert metadata["training_split"] == TRAINING_SPLIT
    assert {f"{variant}:{seed}" for seed, variant, _ in V04_TRAIN_DOCUMENTS} == set(model.training_document_ids)
    validate_model_metadata(model, metadata)


def test_bootstrap_uses_original_document_families_not_120_observations() -> None:
    rows = load_confirmation(ROOT / "artifacts" / "v0.4" / "ablation.json")
    interval = family_bootstrap_interval(rows, draws=100)
    assert set(interval) == {"rule", "learned", "delta"}
    assert interval["delta"][0] < interval["delta"][1]


def test_final_case_manifest_is_tied_to_frozen_final_audit() -> None:
    validate_final_case_manifest(
        ROOT / "artifacts" / "final" / "cases" / "manifest.json",
        ROOT / "artifacts" / "final" / "independent_evaluation.json",
    )
