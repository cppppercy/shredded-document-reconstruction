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


# New V0.4 sources deliberately differ from the two V0.3 OSTP reports.  Each
# record retains a stable landing page and a direct file URL so the experiment
# remains reproducible.  These pages are *only* simulatedly cut for testing.
V04_EXTERNAL_SOURCES = (
    ExternalSource(
        "usgs-landstat-economic-valuation",
        "USGS Open-File Report 2019-1112: Economic valuation of Landsat imagery",
        "https://pubs.usgs.gov/of/2019/1112/ofr20191112.pdf",
        "https://pubs.usgs.gov/publication/ofr20191112",
        "USGS states that its published-report PDFs are public domain; this report notes any exceptions within the text.",
        (2, 8),
    ),
    ExternalSource(
        "nist-digital-identity-guidelines",
        "NIST SP 800-63A-4: Digital Identity Guidelines",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63a-4.pdf",
        "https://csrc.nist.gov/pubs/sp/800/63/a/4/final",
        "NIST states this publication is not subject to copyright in the United States (17 U.S.C. 105).",
        (5, 15),
    ),
    ExternalSource(
        "gpo-annual-report-2023",
        "U.S. Government Publishing Office Annual Report 2023",
        "https://www.govinfo.gov/content/pkg/GOVPUB-GP-d7a27c6b9d380a16190fb4651b2e74da/pdf/GOVPUB-GP-d7a27c6b9d380a16190fb4651b2e74da.pdf",
        "https://www.govinfo.gov/app/details/GOVPUB-GP-d7a27c6b9d380a16190fb4651b2e74da",
        "The GovInfo record states this U.S. Government work is not subject to copyright under 17 U.S.C. 105.",
        (3, 10),
    ),
)


def _download(url: str, destination: Path) -> str:
    request = Request(url, headers={"User-Agent": "shredded-document-reconstruction/0.3 educational project"})
    with urlopen(request, timeout=90) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def source_manifest(sources: tuple[ExternalSource, ...], records: list[dict[str, object]]) -> dict[str, object]:
    """Build provenance for the exact collection of sources just rendered."""
    return {"sources": [asdict(source) for source in sources], "pages": records}


def prepare_pages_for_sources(
    output_dir: Path,
    cache_dir: Path,
    sources: tuple[ExternalSource, ...],
) -> list[dict[str, object]]:
    """Render declared external pages and return their provenance records.

    These are external pages that are *simulatedly cut* later. They are not
    claimed to be photographs of physically shredded documents and never enter
    a training or validation split.
    """
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source in sources:
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
                        # Persist a portable relative path; Windows backslashes
                        # in JSON made manifests awkward to consume elsewhere.
                        "rendered_file": (Path("pages") / filename).as_posix(),
                        "pdf_sha256": sha256,
                        "usage": "external full page, then simulated vertical cutting and shuffling; never model training",
                    }
                )
        finally:
            document.close()
        sleep(0.2)
    (output_dir / "external_sources.json").write_text(json.dumps(source_manifest(sources, records), indent=2), encoding="utf-8")
    return records


def prepare_external_pages(output_dir: Path, cache_dir: Path) -> list[dict[str, object]]:
    """Render the ten V0.3 external pages (kept unchanged for reproducibility)."""
    return prepare_pages_for_sources(output_dir, cache_dir, EXTERNAL_SOURCES)
