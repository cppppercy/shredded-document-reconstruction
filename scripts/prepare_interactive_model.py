"""Build the fixed lightweight model used by the interactive application."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.conditions import CONDITION_NAMES, observation_condition
from shredded_reconstruction.learned import train_v03_compatibility_model
from shredded_reconstruction.v03_splits import V03_TRAIN_DOCUMENTS


def main() -> int:
    output_dir = ROOT / "artifacts" / "final" / "interactive_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = train_v03_compatibility_model(
        V03_TRAIN_DOCUMENTS,
        augmentations=tuple(observation_condition(name) for name in CONDITION_NAMES),
        endpoint_multiplier=0.0,
    )
    joblib.dump(model, output_dir / "compatibility_model.joblib", compress=3)
    (output_dir / "metadata.json").write_text(json.dumps({
        "purpose": "fixed model used only by the Streamlit interactive demo",
        "training_split": "V0.3 training document families; distinct from V0.4 confirmation training",
        "training_document_ids": list(model.training_document_ids),
        "sample_count": model.sample_count,
        "feature_version": model.feature_version,
        "max_vertical_shift": model.max_vertical_shift,
        "endpoint_multiplier": model.endpoint_multiplier,
        "scikit_learn": sklearn.__version__,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
