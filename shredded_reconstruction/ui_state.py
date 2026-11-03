"""Small, testable state transitions for the Streamlit reconstruction UI.

This module deliberately contains no Streamlit code.  It keeps the UI's
prepare/cut/shuffle/predict stages separate, and labels remain unavailable to
the predictors until a prediction has finished.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
import random

from PIL import Image

from .shred import ShreddedDocument, cut_vertical_strips


SOURCE_KEY = "workflow_source_key"
CUT_KEY = "workflow_cut"
SHUFFLED_KEY = "workflow_shuffled"
RESULTS_KEY = "workflow_results"


def clear_after_source_change(state: MutableMapping[str, object], source_key: str) -> bool:
    """Set the active source and invalidate every downstream artefact.

    Returns whether a source replacement occurred.  Uploaded bytes are hashed
    by the caller, so a newly uploaded file cannot retain an old prediction.
    """
    if state.get(SOURCE_KEY) == source_key:
        return False
    state[SOURCE_KEY] = source_key
    clear_after_cut(state)
    return True


def clear_after_cut(state: MutableMapping[str, object]) -> None:
    """Discard cut, shuffled and predicted data after source/count changes."""
    state.pop(CUT_KEY, None)
    state.pop(SHUFFLED_KEY, None)
    state.pop(RESULTS_KEY, None)


def clear_after_shuffle(state: MutableMapping[str, object]) -> None:
    """Discard predictions because their input permutation changed."""
    state.pop(RESULTS_KEY, None)


def prepare_cut(image: Image.Image, strip_count: int) -> ShreddedDocument:
    """Cut a page but keep the visible strips in their original left-right order."""
    strips = tuple(cut_vertical_strips(image.convert("RGB"), strip_count))
    return ShreddedDocument(strips, tuple(range(strip_count)), strip_count)


def shuffle_cut_document(
    cut: ShreddedDocument,
    *,
    rng: random.Random | random.SystemRandom | None = None,
    previous_original_indices: Sequence[int] | None = None,
) -> ShreddedDocument:
    """Create a new shuffled observation without exposing labels to solvers.

    The returned ``original_indices`` stay with the observation solely for
    post-hoc metrics.  We avoid an unchanged order and, when possible, the
    immediately previous order so successive button clicks visibly change.
    """
    rng = rng or random.SystemRandom()
    count = cut.strip_count
    original = tuple(range(count))
    candidate = list(original)
    for _ in range(32):
        rng.shuffle(candidate)
        if tuple(candidate) != original and tuple(candidate) != tuple(previous_original_indices or ()):
            break
    else:
        candidate = list(reversed(original))
    return ShreddedDocument(
        tuple(cut.strips[index] for index in candidate),
        tuple(cut.original_indices[index] for index in candidate),
        count,
    )
