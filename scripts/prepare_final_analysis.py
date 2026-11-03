"""Create reproducible research-dashboard case figures from saved predictions.

No method is rerun here: the orders come from the immutable independent
confirmation CSV/JSON, then are checked against regenerated input strips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.conditions import observation_condition
from shredded_reconstruction.baseline import reconstruct_baseline
from shredded_reconstruction.document import generate_document
from shredded_reconstruction.evaluation import order_metrics
from shredded_reconstruction.learned import train_v03_compatibility_model
from shredded_reconstruction.research_analysis import LEARNED_METHOD, RULE_METHOD, case_categories, load_confirmation
from shredded_reconstruction.shred import compose_strips, shred_document
from shredded_reconstruction.visualize import comparison_figure, shuffled_montage
from shredded_reconstruction.conditions import CONDITION_NAMES
from shredded_reconstruction.v04_splits import V04_TRAIN_DOCUMENTS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts" / "v0.4" / "ablation.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "final" / "cases")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    learned_model = train_v03_compatibility_model(
        V04_TRAIN_DOCUMENTS,
        augmentations=tuple(observation_condition(name) for name in CONDITION_NAMES),
        endpoint_multiplier=0.0,
    )

    manifest: dict[str, object] = {}
    for category, rows in case_categories(load_confirmation(args.input)).items():
        if not rows:
            continue
        row = rows[0]
        seed, variant = int(row["seed"]), str(row["variant"])
        condition = observation_condition(str(row["condition"]), seed=seed)
        original = generate_document(seed=seed, variant=variant, with_border=condition.with_border, with_texture=condition.with_texture)
        shredded = shred_document(original, int(row["strips"]), int(row["shuffle_seed"]), condition.disturbance)
        rule_order, _ = reconstruct_baseline(shredded.strips, max_vertical_shift=3, endpoint_weight=1.0)
        learned_order, _ = learned_model.reconstruct(shredded.strips)
        rule_metrics = order_metrics(rule_order, shredded.original_indices)
        learned_metrics = order_metrics(learned_order, shredded.original_indices)
        if rule_metrics["predicted_original_order"] != row[f"{RULE_METHOD}_order"] or learned_metrics["predicted_original_order"] != row[f"{LEARNED_METHOD}_order"]:
            raise RuntimeError(f"stored prediction is not reproducible for {category}")
        figure = comparison_figure(
            original,
            shuffled_montage(shredded.strips),
            compose_strips(shredded.strips, rule_order),
            compose_strips(shredded.strips, learned_order),
            rule_metrics,
            learned_metrics,
            panel_size=(560, 450),
        )
        output = args.output_dir / f"{category}.png"
        figure.save(output, optimize=True)
        manifest[category] = {
            "source": "artifacts/v0.4/ablation.json confirmation_cases",
            "seed": seed, "variant": variant, "condition": row["condition"], "strips": row["strips"],
            "shuffle_seed": row["shuffle_seed"], "rule_order": rule_order, "learned_order": learned_order,
            "rule_exact": rule_metrics["exact_match"], "learned_exact": learned_metrics["exact_match"],
            "image": output.name,
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
