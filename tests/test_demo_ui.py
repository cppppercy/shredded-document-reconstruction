from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from shredded_reconstruction.visualize import (
    fit_for_display,
    reconstruction_diagnostic,
    strip_workbench_montage,
    wrong_positions,
)


ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_marks_wrong_reconstruction_positions():
    strips = [Image.new("RGB", (12, 16), color) for color in ("red", "green", "blue")]
    order = [1, 0, 2]
    original_indices = [0, 1, 2]
    image = reconstruction_diagnostic(strips, order, original_indices)
    assert image.size == (36, 62)
    assert wrong_positions(order, original_indices) == [1, 2]


def test_display_previews_are_contained_and_strip_montage_is_visibly_separated():
    original = Image.new("RGB", (1200, 2400), "white")
    preview = fit_for_display(original, (520, 430))
    assert preview.size == (215, 430)
    assert original.size == (1200, 2400)

    strips = [Image.new("RGB", (20, 50), color) for color in ("red", "green", "blue")]
    montage = strip_workbench_montage(strips, gap=6)
    # Padding and two gaps make this wider than a seamless reconstruction.
    assert montage.width == 20 * 3 + 20 + 6 * 2
    assert montage.height == 50 + 20


def test_streamlit_page_starts_idle_then_allows_cut_and_shuffle():
    def button(app, key: str):
        return next(item for item in app.button if item.key == key)

    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=60)
    assert not app.exception
    assert any("碎纸文档自动重建" in heading.value for heading in app.title)
    assert button(app, "shuffle_document").disabled
    assert button(app, "reconstruct_document").disabled
    assert app.radio(key="workbench_view_choice").options == ["文档预览"]

    button(app, "cut_document").click()
    app.run(timeout=60)
    assert not app.exception
    assert "裁切结果" in app.radio(key="workbench_view_choice").options
    assert not button(app, "shuffle_document").disabled

    button(app, "shuffle_document").click()
    app.run(timeout=60)
    assert not app.exception
    assert "打乱结果" in app.radio(key="workbench_view_choice").options
    assert not button(app, "reconstruct_document").disabled

    # Selecting a different gallery card is a genuine source replacement, so
    # stale cut/shuffle state must not survive it.
    button(app, "toggle-gallery").click()
    app.run(timeout=60)
    next(item for item in app.button if item.key.startswith("select-")).click()
    app.run(timeout=60)
    assert not app.exception
    assert button(app, "shuffle_document").disabled
    assert button(app, "reconstruct_document").disabled
    assert app.radio(key="workbench_view_choice").options == ["文档预览"]
