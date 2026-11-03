from __future__ import annotations

from shredded_reconstruction.external import ExternalSource, source_manifest


def test_manifest_uses_the_actual_sources_not_a_fixed_global_list() -> None:
    sources = (
        ExternalSource("new-a", "A", "https://example.test/a.pdf", "https://example.test/a", "test", (0,)),
        ExternalSource("new-b", "B", "https://example.test/b.pdf", "https://example.test/b", "test", (1,)),
    )
    records = [
        {"source_id": "new-a", "rendered_file": "pages/a.png"},
        {"source_id": "new-b", "rendered_file": "pages/b.png"},
    ]
    manifest = source_manifest(sources, records)
    assert {entry["source_id"] for entry in manifest["sources"]} == {"new-a", "new-b"}
    assert {entry["source_id"] for entry in manifest["pages"]} == {"new-a", "new-b"}
    assert all("\\" not in entry["rendered_file"] for entry in manifest["pages"])
