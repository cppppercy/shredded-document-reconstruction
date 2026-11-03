from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from shredded_reconstruction.visualize import reconstruction_diagnostic, wrong_positions


ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_marks_wrong_reconstruction_positions():
    strips = [Image.new("RGB", (12, 16), color) for color in ("red", "green", "blue")]
    order = [1, 0, 2]
    original_indices = [0, 1, 2]
    image = reconstruction_diagnostic(strips, order, original_indices)
    assert image.size == (36, 62)
    assert wrong_positions(order, original_indices) == [1, 2]


def test_streamlit_page_starts_idle_then_allows_cut_and_shuffle():
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=60)
    assert not app.exception
    assert any("碎纸文档自动重建" in heading.value for heading in app.title)
    assert not any("重建结果" == heading.value for heading in app.subheader)
    assert app.button[1].disabled
    assert app.button[2].disabled

    app.button[0].click()
    app.run(timeout=60)
    assert not app.exception
    assert any("纸条（保持原顺序）" == heading.value for heading in app.subheader)
    assert not app.button[1].disabled

    app.button[1].click()
    app.run(timeout=60)
    assert not app.exception
    assert any("打乱后的纸条" == heading.value for heading in app.subheader)
    assert not app.button[2].disabled
