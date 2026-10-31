import numpy as np
import pytest
from PIL import Image

from shredded_reconstruction.conditions import CONDITION_NAMES, observation_condition
from shredded_reconstruction.document import generate_document
from shredded_reconstruction.evaluation import run_case
from shredded_reconstruction.learned import train_compatibility_model
from shredded_reconstruction.shred import DisturbanceConfig, shred_document
from shredded_reconstruction.splits import (
    TEST_DOCUMENTS,
    TRAIN_DOCUMENTS,
    VALIDATION_DOCUMENTS,
    assert_disjoint_document_splits,
    document_ids,
)


def test_default_document_and_no_disturbance_preserve_v01_behavior():
    default = generate_document(83, "mixed")
    explicit = generate_document(83, "mixed", with_border=True, with_texture=None)
    assert default.tobytes() == explicit.tobytes()
    old_style = shred_document(default, 8, shuffle_seed=11)
    no_effect = shred_document(default, 8, shuffle_seed=11, disturbance=DisturbanceConfig())
    assert old_style.original_indices == no_effect.original_indices
    assert [item.tobytes() for item in old_style.strips] == [item.tobytes() for item in no_effect.strips]


def test_disturbances_are_reproducible_and_edge_loss_is_permanent():
    image = generate_document(84, "table")
    config = DisturbanceConfig(document_brightness=0.97, blur_radius=0.5, strip_brightness_jitter=0.03, edge_loss=2, vertical_jitter=3, seed=91)
    first = shred_document(image, 10, shuffle_seed=12, disturbance=config)
    second = shred_document(image, 10, shuffle_seed=12, disturbance=config)
    assert first.original_indices == second.original_indices
    assert [strip.tobytes() for strip in first.strips] == [strip.tobytes() for strip in second.strips]
    assert sum(strip.width for strip in first.strips) < image.width
    assert {strip.height for strip in first.strips} == {image.height}


def test_vertical_shift_has_no_wraparound():
    array = np.full((20, 20), 255, dtype=np.uint8)
    array[0, :] = 0
    image = Image.fromarray(array, mode="L")
    # Fixed RNG produces a non-zero shift. A circular roll would show black
    # pixels at both canvas ends; a fresh canvas cannot.
    shredded = shred_document(image, 2, shuffle_seed=1, disturbance=DisturbanceConfig(vertical_jitter=3, seed=0))
    for strip in shredded.strips:
        pixels = np.asarray(strip)
        assert not (np.any(pixels[0] == 0) and np.any(pixels[-1] == 0))


def test_fixed_splits_are_disjoint_by_document_identity():
    assert_disjoint_document_splits(TRAIN_DOCUMENTS, VALIDATION_DOCUMENTS, TEST_DOCUMENTS)
    assert not (document_ids(TRAIN_DOCUMENTS) & document_ids(TEST_DOCUMENTS))
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_document_splits(((17, "report", 8),), ((17, "report", 12),))


def test_all_conditions_yield_legal_predictions():
    for condition in CONDITION_NAMES:
        result = run_case(seed=120, variant="mixed", strip_count=8, shuffle_seed=33, condition_name=condition)
        assert sorted(result.baseline_order) == list(range(8))
        assert 0.0 <= result.baseline_metrics["adjacency_accuracy"] <= 1.0


def test_no_assists_really_removes_v01_page_cues():
    clean = observation_condition("clean", seed=90)
    ablated = observation_condition("no_assists", seed=90)
    assert clean.with_border and clean.with_texture is None
    assert not ablated.with_border and ablated.with_texture is False


def test_legacy_learning_labels_stay_in_source_order():
    """Regression guard for accidentally shuffling labelled training strips."""
    model = train_compatibility_model()
    result = run_case(seed=3001, variant="report", strip_count=8, shuffle_seed=41, model=model)
    assert result.learned_metrics["adjacency_accuracy"] == 1.0
