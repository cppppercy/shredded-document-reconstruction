from __future__ import annotations

from PIL import Image

import shredded_reconstruction.ui_reconstruction as ui_reconstruction
from shredded_reconstruction.shred import ShreddedDocument


class _RecordingLearnedModel:
    def __init__(self) -> None:
        self.input_id: int | None = None

    def reconstruct(self, strips):
        self.input_id = id(strips)
        return [2, 0, 1], 0.0


def test_both_ui_methods_receive_the_same_current_shuffled_tuple(monkeypatch) -> None:
    strips = tuple(Image.new("RGB", (8, 12), color) for color in ("red", "green", "blue"))
    shuffled = ShreddedDocument(strips, (1, 2, 0), 3)
    baseline_input_ids: list[int] = []

    def fake_baseline(input_strips, **_kwargs):
        baseline_input_ids.append(id(input_strips))
        return [2, 0, 1], 0.0

    monkeypatch.setattr(ui_reconstruction, "reconstruct_baseline", fake_baseline)
    learned = _RecordingLearnedModel()
    results = ui_reconstruction.predict_both_methods(shuffled, learned)

    assert baseline_input_ids == [id(shuffled.strips)]
    assert learned.input_id == id(shuffled.strips)
    assert results["rule"]["metrics"]["adjacency_accuracy"] == 1.0
    assert results["learned"]["metrics"]["exact_match"] is True
