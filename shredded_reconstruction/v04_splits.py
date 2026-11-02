"""Independent document-family splits for V0.4 confirmation experiments."""

from __future__ import annotations

from .splits import DocumentSpec, _make_specs


# These document identities are intentionally new: they are separate from the
# V0.1/V0.2 seeds (< 5500) and V0.3 seeds (6100--8312).  The confirmation set
# is never used to choose an ablation or model hyperparameter.
V04_TRAIN_DOCUMENTS: tuple[DocumentSpec, ...] = _make_specs(9100, repetitions=2)       # 24
V04_VALIDATION_DOCUMENTS: tuple[DocumentSpec, ...] = _make_specs(10100, repetitions=1) # 12
V04_CONFIRMATION_DOCUMENTS: tuple[DocumentSpec, ...] = _make_specs(11100, repetitions=2) # 24
