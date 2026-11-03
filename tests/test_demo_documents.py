from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_offline_demo_gallery_contains_ten_readable_documents() -> None:
    directory = ROOT / "demo_documents"
    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    documents = payload["documents"]
    assert len(documents) == 10
    assert {item["kind"] for item in documents} == {"external", "synthetic"}
    assert all((directory / item["filename"]).is_file() for item in documents)
    assert len({item["id"] for item in documents}) == len(documents)


def test_corrected_external_manifest_has_portable_paths_and_matching_sources() -> None:
    directory = ROOT / "artifacts" / "v0.4" / "external"
    payload = json.loads((directory / "external_sources.corrected.json").read_text(encoding="utf-8"))
    source_ids = {source["source_id"] for source in payload["sources"]}
    assert {page["source_id"] for page in payload["pages"]} == source_ids
    assert all("\\" not in page["rendered_file"] for page in payload["pages"])
