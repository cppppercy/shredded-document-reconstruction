"""Create the offline document gallery used by the Streamlit demonstration."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.document import generate_document
from shredded_reconstruction.external import V04_EXTERNAL_SOURCES


DEMO_DIR = ROOT / "demo_documents"
EXTERNAL_DIR = ROOT / "artifacts" / "v0.4" / "external"


def main() -> int:
    DEMO_DIR.mkdir(exist_ok=True)
    manifest = {
        "description": "Offline demo pages. External pages are full pages that may later be simulatedly cut; they are not physical-shred scans.",
        "documents": [],
    }
    source_data = json.loads((EXTERNAL_DIR / "external_sources.json").read_text(encoding="utf-8"))
    # The historical V0.4 manifest was produced before the source-list bug
    # was fixed.  Its page rows are retained as evidence; this gallery uses
    # the declared V0.4 source definitions that produced those rows.
    sources = {source.source_id: source for source in V04_EXTERNAL_SOURCES}
    for page in source_data["pages"]:
        # V0.4's historical evidence used Windows separators.  Normalise only
        # while reading it so this gallery can also be rebuilt on Linux.
        original = EXTERNAL_DIR / Path(str(page["rendered_file"]).replace("\\", "/"))
        filename = f"external-{page['source_id']}-page-{page['page_number_one_based']:02d}.png"
        shutil.copyfile(original, DEMO_DIR / filename)
        source = sources[page["source_id"]]
        manifest["documents"].append(
            {
                "id": filename.removesuffix(".png"), "filename": filename,
                "title": f"{source.title}（第 {page['page_number_one_based']} 页）",
                "kind": "external", "source_id": source.source_id,
                "source_page_url": source.source_page_url, "license_note": source.license_note,
            }
        )
    for seed, variant, title in (
        (12001, "report", "合成示例：文字报告"),
        (12002, "table", "合成示例：数据表格"),
        (12003, "mixed", "合成示例：图文混排"),
        (12004, "sparse", "合成示例：稀疏版式"),
    ):
        filename = f"synthetic-{variant}.png"
        generate_document(seed=seed, variant=variant).save(DEMO_DIR / filename, optimize=True)
        manifest["documents"].append(
            {"id": filename.removesuffix(".png"), "filename": filename, "title": title, "kind": "synthetic", "source_id": None,
             "source_page_url": None, "license_note": "Project-generated demonstration page."}
        )
    (DEMO_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (DEMO_DIR / "README.md").write_text(
        "# 演示文档\n\n本目录是离线 Streamlit 演示图库。外部页面的来源和许可记录在 `manifest.json`；它们仅被模拟裁切，不是物理碎纸扫描。四张合成页面由项目代码生成。\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(manifest['documents'])} offline demo documents in {DEMO_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
