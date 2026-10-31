"""Fixed document-family splits used by V0.2, independent of perturbations."""

from __future__ import annotations

from collections.abc import Iterable

from .document import DOCUMENT_VARIANTS
from .learned import document_identity

DocumentSpec = tuple[int, str, int]


def _make_specs(start_seed: int, repetitions: int) -> tuple[DocumentSpec, ...]:
    specs: list[DocumentSpec] = []
    for repeat in range(repetitions):
        for variant_index, variant in enumerate(DOCUMENT_VARIANTS):
            for strip_index, strip_count in enumerate((8, 10, 12)):
                seed = start_seed + repeat * 100 + variant_index * 10 + strip_index
                specs.append((seed, variant, strip_count))
    return tuple(specs)


# Split by original document. Every condition is a derived observation of only
# one source family, never a separate train/validation/test document.
TRAIN_DOCUMENTS = _make_specs(1100, repetitions=3)       # 36 families
VALIDATION_DOCUMENTS = _make_specs(4100, repetitions=1)  # 12 families
TEST_DOCUMENTS = _make_specs(5100, repetitions=4)        # 48 families


def document_ids(specs: Iterable[DocumentSpec]) -> set[str]:
    return {document_identity(seed, variant) for seed, variant, _ in specs}


def assert_disjoint_document_splits(*splits: Iterable[DocumentSpec]) -> None:
    """Fail on actual source identity overlap, never fragile string matching."""
    seen: set[str] = set()
    for specs in splits:
        identities = document_ids(specs)
        overlap = seen & identities
        if overlap:
            raise ValueError(f"document-family leakage across splits: {sorted(overlap)}")
        seen |= identities
