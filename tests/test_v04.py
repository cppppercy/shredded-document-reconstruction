from __future__ import annotations

from shredded_reconstruction.splits import (
    TEST_DOCUMENTS,
    TRAIN_DOCUMENTS,
    VALIDATION_DOCUMENTS,
    assert_disjoint_document_splits,
    document_ids,
)
from shredded_reconstruction.v03_splits import V03_TEST_DOCUMENTS, V03_TRAIN_DOCUMENTS, V03_VALIDATION_DOCUMENTS
from shredded_reconstruction.v04_splits import V04_CONFIRMATION_DOCUMENTS, V04_TRAIN_DOCUMENTS, V04_VALIDATION_DOCUMENTS


def test_v04_document_families_are_new_and_disjoint() -> None:
    assert_disjoint_document_splits(
        TRAIN_DOCUMENTS,
        VALIDATION_DOCUMENTS,
        TEST_DOCUMENTS,
        V03_TRAIN_DOCUMENTS,
        V03_VALIDATION_DOCUMENTS,
        V03_TEST_DOCUMENTS,
        V04_TRAIN_DOCUMENTS,
        V04_VALIDATION_DOCUMENTS,
        V04_CONFIRMATION_DOCUMENTS,
    )
    assert len(document_ids(V04_CONFIRMATION_DOCUMENTS)) == 24
