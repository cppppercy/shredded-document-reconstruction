"""Write a corrected companion manifest without overwriting V0.4 evidence."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.external import V04_EXTERNAL_SOURCES


def main() -> int:
    directory = ROOT / "artifacts" / "v0.4" / "external"
    original_path = directory / "external_sources.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    expected = {source.source_id for source in V04_EXTERNAL_SOURCES}
    page_ids = {page["source_id"] for page in original["pages"]}
    if page_ids != expected:
        raise ValueError(f"Unexpected historical V0.4 page sources: {page_ids}")
    corrected = {
        "correction": {
            "date": "2026-11-03",
            "reason": "The original top-level sources field was written from a fixed V0.3 list. Page-level records were correct and are preserved in the original manifest.",
            "original_manifest": "external_sources.json",
            "portable_rendered_paths": "Existing historical paths are preserved; future manifests use POSIX relative paths.",
        },
        "sources": [asdict(source) for source in V04_EXTERNAL_SOURCES],
        "pages": [
            {**page, "rendered_file": str(page["rendered_file"]).replace("\\", "/")}
            for page in original["pages"]
        ],
    }
    output = directory / "external_sources.corrected.json"
    output.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
