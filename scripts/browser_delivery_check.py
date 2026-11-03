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
from urllib.parse import urlparse

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


def _state(page) -> dict[str, str]:
    token = page.evaluate(
        """() => (document.body.textContent || '').match(/run-(\\d+)/)?.[1] || null"""
    )
    if token is None:
        raise RuntimeError("workflow completion token was not rendered")
    return {"revision": token}


def _wait_for_images(page) -> None:
    """Wait for the browser to finish rendering currently supplied images."""
    page.wait_for_function(
        """() => Array.from(document.images).every((image) =>
            image.complete && image.naturalWidth > 0)""",
        timeout=30_000,
    )


def _wait_for_workflow(page, *, stage: str, after_revision: int | None = None, source_kind: str | None = None) -> dict[str, str]:
    """Wait for an actual Streamlit result, not a coincidental widget count."""
    page.wait_for_function(
        """({afterRevision, sourceKind, stage}) => {
            const match = (document.body.textContent || '').match(/run-(\\d+)/);
            if (!match || (afterRevision !== null && Number(match[1]) <= afterRevision)) return false;
            if (!(document.body.textContent || '').includes(`stage=${stage}`)) return false;
            if (sourceKind === 'upload' && !(document.body.textContent || '').includes('large-page.png')) return false;
            if (sourceKind === 'demo' && document.querySelector('input[type="file"]')) return false;
            return true;
        }""",
        arg={"afterRevision": after_revision, "sourceKind": source_kind, "stage": stage},
        timeout=90_000,
    )
    _wait_for_images(page)
    # A short quiet period catches a stale DOM node that was replaced while a
    # Streamlit rerun was still applying its final deltas.
    first = _state(page)
    for _ in range(5):
        page.wait_for_timeout(150)
        current = _state(page)
        if current == first:
            return current
        first = current
    return first


def _wait_for_analysis(page) -> None:
    page.locator('[data-testid="stDataFrame"]').first.wait_for(timeout=30_000)
    _wait_for_images(page)
    page.wait_for_timeout(300)


def _resource_summary(page) -> dict[str, float | int]:
    entries = page.evaluate(
        """() => performance.getEntriesByType('resource').map((entry) => ({
            name: entry.name, duration: entry.duration, transferSize: entry.transferSize
        }))"""
    )
    remote_http_hosts = sorted(
        {
            parsed.hostname
            for entry in entries
            if (parsed := urlparse(str(entry.get("name", "")))).scheme in {"http", "https"}
            and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        }
    )
    return {
        "resources": len(entries),
        "transfer_bytes": sum(int(entry.get("transferSize", 0)) for entry in entries),
        "slowest_resource_seconds": max((float(entry.get("duration", 0)) for entry in entries), default=0.0) / 1000,
        "external_http_hosts": len(remote_http_hosts),
    }


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
    resources: dict[str, dict[str, float | int]] = {}
    trace: list[str] = []
    try:
        _wait_for_server(url)
        trace.append("server-ready")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=str(_chrome_path(args.chrome_path)), headless=True)
            trace.append("chrome-ready")
            page = browser.new_page(viewport={"width": 1366, "height": 768}, device_scale_factor=1)

            started = perf_counter()
            page.goto(url, wait_until="domcontentloaded")
            _wait_for_workflow(page, stage="source", source_kind="demo")
            timings["cold_open"] = perf_counter() - started
            resources["cold_open"] = _resource_summary(page)
            trace.append("cold-open")

            started = perf_counter()
            page.reload(wait_until="domcontentloaded")
            _wait_for_workflow(page, stage="source", source_kind="demo")
            timings["warm_open"] = perf_counter() - started
            resources["warm_open"] = _resource_summary(page)
            trace.append("warm-open")

            started = perf_counter()
            page.locator('[data-testid="stRadio"]').nth(1).locator("input").nth(1).check(force=True)
            upload = page.locator('input[type="file"]')
            upload.wait_for(state="attached", timeout=30_000)
            upload.set_input_files({"name": "large-page.png", "mimeType": "image/png", "buffer": _large_png()})
            upload_state = _wait_for_workflow(page, stage="source", source_kind="upload")
            timings["large_upload"] = perf_counter() - started
            resources["large_upload"] = _resource_summary(page)
            trace.append("large-upload")

            started = perf_counter()
            page.locator('[data-testid="stRadio"]').nth(1).locator("input").nth(0).check(force=True)
            source_state = _wait_for_workflow(
                page, stage="source", source_kind="demo", after_revision=int(upload_state["revision"])
            )
            timings["upload_to_demo"] = perf_counter() - started
            resources["upload_to_demo"] = _resource_summary(page)
            trace.append("upload-to-demo")

            started = perf_counter()
            page.locator('[data-testid="stButton"] button').nth(1).click()
            cut_state = _wait_for_workflow(page, stage="cut", after_revision=int(source_state["revision"]))
            timings["cut"] = perf_counter() - started
            trace.append("cut")

            started = perf_counter()
            page.locator('[data-testid="stButton"] button').nth(2).click()
            shuffled_state = _wait_for_workflow(page, stage="shuffled", after_revision=int(cut_state["revision"]))
            timings["shuffle"] = perf_counter() - started
            trace.append("shuffle")

            started = perf_counter()
            page.locator('[data-testid="stButton"] button').nth(3).click()
            first_results = _wait_for_workflow(page, stage="results", after_revision=int(shuffled_state["revision"]))
            timings["first_reconstruction"] = perf_counter() - started
            trace.append("first-reconstruction")
            page.screenshot(path=str(args.output_dir / "results-1366.png"), full_page=False)

            started = perf_counter()
            page.locator('[data-testid="stButton"] button').nth(3).click()
            _wait_for_workflow(page, stage="results", after_revision=int(first_results["revision"]))
            timings["repeat_reconstruction"] = perf_counter() - started
            trace.append("repeat-reconstruction")

            started = perf_counter()
            page.locator('[data-testid="stRadio"]').nth(0).locator("input").nth(1).check(force=True)
            _wait_for_analysis(page)
            timings["analysis_switch"] = perf_counter() - started
            resources["analysis_switch"] = _resource_summary(page)
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
        "resource_summary": resources,
        "notes": "Each value is a browser-driven end-to-end action until a stage-specific Streamlit completion marker is emitted, all current images have loaded, and the marker stays stable. Resource summaries count distinct non-loopback HTTP(S) hosts so the local demo can be checked for runtime remote fetches. It includes local browser and server work, but is not a cross-machine benchmark.",
    }
    (args.output_dir / "timings.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
