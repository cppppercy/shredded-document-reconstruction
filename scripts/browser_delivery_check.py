"""Exercise the real Streamlit UI in installed Chrome and save timing evidence.

This is deliberately separate from AppTest: it measures a browser-driven
workflow (including upload, reruns and screenshots).  It uses a locally
installed Chrome rather than downloading a Playwright browser.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from subprocess import DEVNULL, Popen
import sys
from time import perf_counter, sleep

import requests
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def _chrome_path(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise FileNotFoundError("Chrome was not found; pass --chrome-path to an installed Chrome executable.")


def _wait_for_server(url: str, timeout_seconds: float = 30.0) -> None:
    deadline = perf_counter() + timeout_seconds
    while perf_counter() < deadline:
        try:
            if requests.get(url, timeout=1).ok:
                return
        except requests.RequestException:
            sleep(0.25)
    raise TimeoutError(f"Streamlit did not start at {url}")


def _large_png() -> bytes:
    # 12 MP stays below the UI's 24 MP safety cap while exercising decode and
    # preview containment with a realistic high-resolution upload.
    image = Image.new("RGB", (3000, 4000), "white")
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8516)
    parser.add_argument("--url", help="Use an already-running local Streamlit URL instead of starting a server.")
    parser.add_argument("--chrome-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "final" / "browser_check")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    url = args.url or f"http://127.0.0.1:{args.port}"
    server = None
    if args.url is None:
        server = Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(args.port)],
            cwd=ROOT,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
    timings: dict[str, float] = {}
    trace: list[str] = []
    try:
        _wait_for_server(url)
        trace.append("server-ready")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=str(_chrome_path(args.chrome_path)), headless=True)
            trace.append("chrome-ready")
            page = browser.new_page(viewport={"width": 1366, "height": 768}, device_scale_factor=1)

            started = perf_counter()
            page.goto(url, wait_until="networkidle")
            page.locator('[data-testid="stButton"] button').nth(1).wait_for(timeout=30_000)
            timings["cold_open"] = perf_counter() - started
            trace.append("cold-open")

            started = perf_counter()
            page.reload(wait_until="networkidle")
            page.locator('[data-testid="stButton"] button').nth(1).wait_for(timeout=30_000)
            timings["warm_open"] = perf_counter() - started
            trace.append("warm-open")

            started = perf_counter()
            radios = page.locator('[data-testid="stRadio"] input')
            radios.nth(3).check(force=True)
            upload = page.locator('input[type="file"]')
            upload.wait_for(state="attached", timeout=30_000)
            upload.set_input_files({"name": "large-page.png", "mimeType": "image/png", "buffer": _large_png()})
            page.locator('[data-testid="stButton"] button').nth(1).wait_for(timeout=30_000)
            timings["large_upload"] = perf_counter() - started
            trace.append("large-upload")

            started = perf_counter()
            page.get_by_text("演示文档", exact=True).click()
            page.locator('[data-testid="stButton"] button').nth(1).wait_for(state="visible", timeout=30_000)
            page.locator('[data-testid="stButton"] button').nth(1).wait_for(state="attached", timeout=30_000)
            page.locator('[data-testid="stButton"] button').nth(1).is_enabled()
            page.wait_for_function("document.querySelectorAll('[data-testid=stButton] button')[1].disabled === false", timeout=30_000)
            timings["upload_to_demo"] = perf_counter() - started
            trace.append("upload-to-demo")

            started = perf_counter()
            page.locator('[data-testid="stButton"] button').nth(1).click()
            page.wait_for_function("Math.max(0, document.querySelectorAll('[data-testid=stRadio] input').length - 5)")
            timings["cut"] = perf_counter() - started
            trace.append("cut")

            started = perf_counter()
            page.locator('[data-testid="stButton"] button').nth(2).click()
            page.wait_for_function("Math.max(0, document.querySelectorAll('[data-testid=stRadio] input').length - 6)")
            timings["shuffle"] = perf_counter() - started
            trace.append("shuffle")

            started = perf_counter()
            page.locator('[data-testid="stButton"] button').nth(3).click()
            page.wait_for_function("Math.max(0, document.querySelectorAll('[data-testid=stRadio] input').length - 10)", timeout=60_000)
            page.wait_for_timeout(3_000)
            timings["first_reconstruction"] = perf_counter() - started
            trace.append("first-reconstruction")
            page.screenshot(path=str(args.output_dir / "results-1366.png"), full_page=False)

            started = perf_counter()
            page.locator('[data-testid="stButton"] button').nth(3).click()
            page.wait_for_function("Math.max(0, document.querySelectorAll('[data-testid=stRadio] input').length - 10)", timeout=60_000)
            timings["repeat_reconstruction"] = perf_counter() - started
            trace.append("repeat-reconstruction")

            started = perf_counter()
            page.locator('[data-testid="stRadio"] input').nth(1).check(force=True)
            page.locator('[data-testid="stDataFrame"]').first.wait_for(timeout=30_000)
            page.wait_for_timeout(1_000)
            timings["analysis_switch"] = perf_counter() - started
            trace.append("analysis-switch")
            page.screenshot(path=str(args.output_dir / "analysis-1366.png"), full_page=False)
            browser.close()
    except Exception as error:
        trace.append(f"error: {type(error).__name__}: {error}")
        raise
    finally:
        (args.output_dir / "trace.log").write_text("\n".join(trace), encoding="utf-8")
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except Exception:
                server.kill()
    output = {
        "environment": {"browser": "local Chrome via Playwright", "viewport": "1366x768", "network": "local loopback"},
        "timings_seconds": timings,
        "notes": "Each value is a browser-driven end-to-end action until the expected Streamlit DOM state appears. It includes local browser and server work, but is not a cross-machine benchmark.",
    }
    (args.output_dir / "timings.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
