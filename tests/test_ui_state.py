from __future__ import annotations

import random

from PIL import Image

from shredded_reconstruction.ui_state import (
    CUT_KEY, RESULTS_KEY, SHUFFLED_KEY, VIEW_KEY, clear_after_cut, clear_after_shuffle,
    clear_after_source_change, prepare_cut, shuffle_cut_document,
)


def _page() -> Image.Image:
    image = Image.new("RGB", (80, 40), "white")
    for column in range(80):
        image.putpixel((column, 0), (column * 3, 10, 200 - column * 2))
    return image


def test_source_change_invalidates_every_downstream_stage() -> None:
    state: dict[str, object] = {}
    assert clear_after_source_change(state, "demo:first")
    state[CUT_KEY] = prepare_cut(_page(), 8)
    state[SHUFFLED_KEY] = object()
    state[RESULTS_KEY] = object()
    state[VIEW_KEY] = "恢复结果"
    assert clear_after_source_change(state, "upload:new-bytes")
    assert CUT_KEY not in state and SHUFFLED_KEY not in state and RESULTS_KEY not in state and VIEW_KEY not in state
    assert not clear_after_source_change(state, "upload:new-bytes")


def test_cut_and_shuffle_invalidate_old_predictions() -> None:
    state: dict[str, object] = {RESULTS_KEY: object()}
    cut = prepare_cut(_page(), 8)
    state[CUT_KEY] = cut
    clear_after_cut(state)
    assert CUT_KEY not in state and RESULTS_KEY not in state
    first = shuffle_cut_document(cut, rng=random.Random(4))
    state[SHUFFLED_KEY] = first
    state[RESULTS_KEY] = object()
    state[VIEW_KEY] = "恢复结果"
    clear_after_shuffle(state)
    assert SHUFFLED_KEY in state and RESULTS_KEY not in state and VIEW_KEY not in state
    second = shuffle_cut_document(cut, rng=random.Random(9), previous_original_indices=first.original_indices)
    assert sorted(second.original_indices) == list(range(8))
    assert second.original_indices != first.original_indices
    assert second.original_indices != tuple(range(8))
