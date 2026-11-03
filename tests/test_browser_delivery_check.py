from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_browser_delivery_check_uses_explicit_completion_not_widget_counts():
    source = (ROOT / "scripts" / "browser_delivery_check.py").read_text(encoding="utf-8")
    assert "Math.max(0, document.querySelectorAll" not in source
    # The browser check waits for the explicit workflow revision rendered by
    # the application, rather than treating the number of radio widgets as a
    # completed Streamlit interaction.  Avoid coupling this test to Python's
    # escaped representation of the JavaScript regular expression.
    assert "run-" in source
    assert "afterRevision" in source
    assert "stage=${stage}" in source
    assert "_wait_for_images(page)" in source
    assert "external_http_hosts" in source
    assert "after_revision=int(first_results[\"revision\"])" in source
