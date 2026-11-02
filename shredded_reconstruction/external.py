"""Download and render reproducible, public-domain external document pages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen

import pymupdf as fitz
from PIL import Image


@dataclass(frozen=True)
class ExternalSource:
    source_id: str
    title: str
    pdf_url: str
    source_page_url: str
    license_note: str
    pages_zero_based: tuple[int, ...]


# Each linked GovInfo PDF says that it is a work of the United States
# Government and is in the public domain under 17 U.S.C. §105.  We retain the
# direct PDF URL and human-readable provenance page in the run manifest.
EXTERNAL_SOURCES = (
    ExternalSource(
        "us-cbdc-technical-evaluation",
        "Technical Evaluation for a U.S. CBDC System (OSTP, 2022)",
        "https://www.govinfo.gov/content/pkg/GOVPUB-PREX23-PURL-gpo193426/pdf/GOVPUB-PREX23-PURL-gpo193426.pdf",
        "https://www.govinfo.gov/content/pkg/GOVPUB-PREX23-PURL-gpo193426/pdf/GOVPUB-PREX23-PURL-gpo193426.pdf",
        "The PDF states it is a work of the U.S. Government in the public domain (17 U.S.C. §105).",
        (1, 2, 3, 4, 5),
    ),
    ExternalSource(
        "ai-bill-of-rights",
        "Blueprint for an AI Bill of Rights (OSTP, 2022)",
        "https://www.govinfo.gov/content/pkg/GOVPUB-PREX23-PURL-gpo193638/pdf/GOVPUB-PREX23-PURL-gpo193638.pdf",
        "https://www.govinfo.gov/content/pkg/GOVPUB-PREX23-PURL-gpo193638/pdf/GOVPUB-PREX23-PURL-gpo193638.pdf",
        "The PDF states it is a work of the U.S. Government in the public domain (17 U.S.C. §105).",
        (1, 2, 3, 4, 5),
    ),
)


def _download(url: str, destination: Path) -> str:
    request = Request(url, headers={"User-Agent": "shredded-document-reconstruction/0.3 educational project"})
    with urlopen(request, timeout=90) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def prepare_external_pages(output_dir: Path, cache_dir: Path) -> list[dict[str, object]]:
    """Render ten external pages and return their provenance records.

    These are external pages that are *simulatedly cut* later. They are not
    claimed to be photographs of physically shredded documents and never enter
    a training or validation split.
    """
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source in EXTERNAL_SOURCES:
        pdf_path = cache_dir / f"{source.source_id}.pdf"
        sha256 = _download(source.pdf_url, pdf_path) if not pdf_path.exists() else hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        document = fitz.open(pdf_path)
        try:
            for page_index in source.pages_zero_based:
                if page_index >= document.page_count:
                    raise RuntimeError(f"{source.source_id} has only {document.page_count} pages")
                filename = f"{source.source_id}-page-{page_index + 1:02d}.png"
                page_path = pages_dir / filename
                if not page_path.exists():
                    pixmap = document.load_page(page_index).get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
                    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    image.save(page_path, optimize=True)
                records.append(
                    {
                        "page_id": f"{source.source_id}:{page_index + 1}",
                        "source_id": source.source_id,
                        "title": source.title,
                        "source_pdf_url": source.pdf_url,
                        "source_page_url": source.source_page_url,
                        "license_note": source.license_note,
                        "page_number_one_based": page_index + 1,
                        "rendered_file": str(Path("pages") / filename),
                        "pdf_sha256": sha256,
                        "usage": "external full page, then simulated vertical cutting and shuffling; never model training",
                    }
                )
        finally:
            document.close()
        sleep(0.2)
    (output_dir / "external_sources.json").write_text(
        json.dumps({"sources": [asdict(source) for source in EXTERNAL_SOURCES], "pages": records}, indent=2),
        encoding="utf-8",
    )
    return records
