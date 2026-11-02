import numpy as np

from shredded_reconstruction.conditions import observation_condition
from shredded_reconstruction.document import generate_document
from shredded_reconstruction.evaluation import run_image_case
from shredded_reconstruction.features import V03_FEATURE_NAMES, v03_compatibility_features
from shredded_reconstruction.learned import train_v03_compatibility_model
from shredded_reconstruction.shred import cut_vertical_strips
from shredded_reconstruction.splits import TEST_DOCUMENTS, TRAIN_DOCUMENTS, VALIDATION_DOCUMENTS, assert_disjoint_document_splits
from shredded_reconstruction.v03_splits import V03_TEST_DOCUMENTS, V03_TRAIN_DOCUMENTS, V03_VALIDATION_DOCUMENTS


def test_v03_splits_are_disjoint_from_v02_and_each_other():
    assert_disjoint_document_splits(
        TRAIN_DOCUMENTS, VALIDATION_DOCUMENTS, TEST_DOCUMENTS,
        V03_TRAIN_DOCUMENTS, V03_VALIDATION_DOCUMENTS, V03_TEST_DOCUMENTS,
    )


def test_v03_alignment_features_are_finite_and_label_free():
    image = generate_document(seed=6110, variant="mixed")
    left, right = cut_vertical_strips(image, 8)[:2]
    values = v03_compatibility_features(left, right, max_vertical_shift=3)
    assert values.shape == (len(V03_FEATURE_NAMES),)
    assert np.isfinite(values).all()
    assert 0.0 <= values[-3] <= 1.0
    assert 0.0 <= values[-2] <= 1.0


def test_v03_model_and_uploaded_image_path_return_legal_orders():
    model = train_v03_compatibility_model(
        V03_TRAIN_DOCUMENTS[:1],
        augmentations=(observation_condition("vertical_shift"),),
        endpoint_multiplier=0.0,
    )
    image = generate_document(seed=8110, variant="table")
    case = run_image_case(image, strip_count=8, shuffle_seed=37, model=model, condition_name="scan", baseline_max_vertical_shift=3)
    assert sorted(case.baseline_order) == list(range(8))
    assert sorted(case.learned_order) == list(range(8))
