"""Fixed document-family protocol introduced for the V0.3 experiment."""

from __future__ import annotations

from .splits import DocumentSpec, _make_specs

# These seeds are disjoint from every V0.2 split (which ends below 5500).
# Conditions are derived observations of one family, never new families.
V03_TRAIN_DOCUMENTS: tuple[DocumentSpec, ...] = _make_specs(6100, repetitions=2)       # 24
V03_VALIDATION_DOCUMENTS: tuple[DocumentSpec, ...] = _make_specs(7100, repetitions=2)  # 24
V03_TEST_DOCUMENTS: tuple[DocumentSpec, ...] = _make_specs(8100, repetitions=2)        # 24
