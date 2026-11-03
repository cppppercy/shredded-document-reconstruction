"""Record server-side Streamlit rerun timings for reproducible performance checks."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def _button(app: AppTest, key: str):
    return next(item for item in app.button if item.key == key)


def _run(app: AppTest, label: str, action=None) -> float:
    if action is not None:
        action()
    started = perf_counter()
    app.run(timeout=120)
    if app.exception:
        raise RuntimeError(f"{label}: {app.exception}")
    return perf_counter() - started


def main() -> int:
    app = AppTest.from_file(str(ROOT / "app.py"))
    times = {"first_open": _run(app, "first_open")}
    times["open_gallery"] = _run(app, "open_gallery", lambda: _button(app, "toggle-gallery").click())
    # The first selectable gallery card is a genuine different source.
    times["select_other_demo"] = _run(app, "select_other_demo", lambda: next(item for item in app.button if item.key.startswith("select-")).click())
    source = next(item for item in app.radio if item.label == "文档来源")
    times["demo_to_upload"] = _run(app, "demo_to_upload", lambda: source.set_value("上传图片"))
    times["upload_to_demo"] = _run(app, "upload_to_demo", lambda: next(item for item in app.radio if item.label == "文档来源").set_value("演示文档"))
    navigation = next(item for item in app.radio if item.label == "页面")
    times["switch_analysis"] = _run(app, "switch_analysis", lambda: navigation.set_value("实验分析"))
    times["back_to_workbench"] = _run(app, "back_to_workbench", lambda: next(item for item in app.radio if item.label == "页面").set_value("自动重建"))
    times["cut"] = _run(app, "cut", lambda: _button(app, "cut_document").click())
    times["shuffle"] = _run(app, "shuffle", lambda: _button(app, "shuffle_document").click())
    times["first_reconstruction"] = _run(app, "first_reconstruction", lambda: _button(app, "reconstruct_document").click())
    view = next(item for item in app.radio if item.key == "workbench_view_choice")
    times["result_view_switch"] = _run(app, "result_view_switch", lambda: view.set_value("打乱结果"))
    output = ROOT / "artifacts" / "final" / "streamlit_rerun_performance.json"
    output.write_text(json.dumps({"scope": "server-side Streamlit rerun time; excludes browser paint", "seconds": times}, indent=2), encoding="utf-8")
    print(json.dumps(times, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
