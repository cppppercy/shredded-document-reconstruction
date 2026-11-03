"""Versioned, inspectable model artefacts used by the demo and final audit.

The project deliberately keeps model construction in one place.  This prevents
the interactive demo from silently using a different training split than the
headline independent experiment.
"""

from __future__ import annotations

import platform
from typing import Any

import sklearn

from .conditions import CONDITION_NAMES, observation_condition
from .learned import CompatibilityModel, train_v03_compatibility_model
from .v04_splits import V04_TRAIN_DOCUMENTS


MODEL_ARTIFACT_FORMAT = 1
TRAINING_SPLIT = "V0.4 training document families (24 identities, seeds 9100+ only)"


def train_final_demo_model() -> CompatibilityModel:
    """Build the exact fixed learned configuration used by the final audit.

    This function must not receive an evaluation split.  The final independent
    audit and Streamlit demo both call it (the latter normally loads its saved
    result), so their model provenance stays identical.
    """
    return train_v03_compatibility_model(
        V04_TRAIN_DOCUMENTS,
        augmentations=tuple(observation_condition(name) for name in CONDITION_NAMES),
        endpoint_multiplier=0.0,
    )


def model_metadata(model: CompatibilityModel) -> dict[str, Any]:
    """Return JSON-safe provenance and compatibility facts for ``model``."""
    return {
        "artifact_format": MODEL_ARTIFACT_FORMAT,
        "purpose": "fixed learned model shared by the Streamlit demo and final independent evaluation",
        "training_split": TRAINING_SPLIT,
        "training_document_ids": list(model.training_document_ids),
        "sample_count": model.sample_count,
        "feature_version": model.feature_version,
        "max_vertical_shift": model.max_vertical_shift,
        "endpoint_multiplier": model.endpoint_multiplier,
        "scikit_learn": sklearn.__version__,
        "python": platform.python_version(),
    }


def validate_model_metadata(model: CompatibilityModel, metadata: dict[str, Any]) -> None:
    """Fail closed when a serialized model and its declared provenance differ."""
    required = {
        "artifact_format": MODEL_ARTIFACT_FORMAT,
        "feature_version": model.feature_version,
        "max_vertical_shift": model.max_vertical_shift,
        "endpoint_multiplier": model.endpoint_multiplier,
        "training_document_ids": list(model.training_document_ids),
        "scikit_learn": sklearn.__version__,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Serialized learned model metadata mismatch for {key}: "
                f"expected {expected!r}, got {metadata.get(key)!r}. "
                "Rebuild it with scripts/prepare_interactive_model.py under the locked environment."
            )
    if metadata.get("training_split") != TRAINING_SPLIT:
        raise RuntimeError("Serialized learned model uses an unexpected training split.")
