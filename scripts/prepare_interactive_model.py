"""Build the fixed lightweight model used by the interactive application."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.model_artifact import model_metadata, train_final_demo_model


def main() -> int:
    output_dir = ROOT / "artifacts" / "final" / "interactive_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = train_final_demo_model()
    joblib.dump(model, output_dir / "compatibility_model.joblib", compress=3)
    (output_dir / "metadata.json").write_text(json.dumps(model_metadata(model), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
