"""A final untouched document-family test split for the project audit."""

from __future__ import annotations

from .splits import DocumentSpec, _make_specs


# Kept separate from all previous V0.1--V0.4 ranges.  No implementation or
# parameter is selected after this split is evaluated.
FINAL_AUDIT_DOCUMENTS: tuple[DocumentSpec, ...] = _make_specs(12100, repetitions=2)
