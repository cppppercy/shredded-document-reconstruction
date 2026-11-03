"""Create reproducible final-dashboard cases from the frozen final audit.

The default input is deliberately the *final* independent audit, never a
historical V0.4 ablation. Every output records a content hash of that input,
the exact case row, stored predictions, and a deterministic replay check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.baseline import reconstruct_baseline
from shredded_reconstruction.conditions import observation_condition
from shredded_reconstruction.document import generate_document
from shredded_reconstruction.evaluation import order_metrics
from shredded_reconstruction.model_artifact import model_metadata, train_final_demo_model
from shredded_reconstruction.research_analysis import LEARNED_METHOD, RULE_METHOD, case_categories, load_confirmation
from shredded_reconstruction.shred import compose_strips, shred_document
from shredded_reconstruction.visualize import comparison_figure, shuffled_montage


FINAL_INPUT = ROOT / "artifacts" / "final" / "independent_evaluation.json"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_id(row: dict[str, object]) -> str:
    return f"{row['family_id']}:{row['condition']}:shuffle-{row['shuffle_seed']}"


def _require_final_audit(data: dict[str, object], path: Path) -> None:
    protocol = data.get("protocol", {})
    if protocol.get("purpose") != "final untouched audit; methods and parameters were fixed before this test":
        raise ValueError(
            f"{_relative(path)} is not the frozen final audit. "
            "Historical V0.4 data is protected from overwriting final cases."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FINAL_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "final" / "cases")
    args = parser.parse_args()
    input_path = args.input.resolve()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    _require_final_audit(data, input_path)
    rows = load_confirmation(input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    learned_model = train_final_demo_model()

    cases: dict[str, object] = {}
    for category, category_rows in case_categories(rows).items():
        if not category_rows:
            continue
        row = category_rows[0]
        seed, variant = int(row["seed"]), str(row["variant"])
        condition = observation_condition(str(row["condition"]), seed=seed)
        original = generate_document(
            seed=seed,
            variant=variant,
            with_border=condition.with_border,
            with_texture=condition.with_texture,
        )
        shredded = shred_document(original, int(row["strips"]), int(row["shuffle_seed"]), condition.disturbance)
        rule_order, _ = reconstruct_baseline(shredded.strips, max_vertical_shift=3, endpoint_weight=1.0)
        learned_order, _ = learned_model.reconstruct(shredded.strips)
        rule_metrics = order_metrics(rule_order, shredded.original_indices)
        learned_metrics = order_metrics(learned_order, shredded.original_indices)
        stored_rule = row[f"{RULE_METHOD}_order"]
        stored_learned = row[f"{LEARNED_METHOD}_order"]
        if rule_metrics["predicted_original_order"] != stored_rule or learned_metrics["predicted_original_order"] != stored_learned:
            raise RuntimeError(f"stored prediction is not reproducible for {category} ({_row_id(row)})")
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
        cases[category] = {
            "source_row_id": _row_id(row),
            "seed": seed,
            "variant": variant,
            "condition": row["condition"],
            "strips": row["strips"],
            "shuffle_seed": row["shuffle_seed"],
            "stored_predicted_original_orders": {"rule": stored_rule, "learned": stored_learned},
            "replayed_input_orders": {"rule": rule_order, "learned": learned_order},
            "replayed_predicted_original_orders": {
                "rule": rule_metrics["predicted_original_order"],
                "learned": learned_metrics["predicted_original_order"],
            },
            "rule_exact": rule_metrics["exact_match"],
            "learned_exact": learned_metrics["exact_match"],
            "image": output.name,
            "image_sha256": _sha256(output),
        }
    manifest = {
        "schema_version": 2,
        "input": {
            "path": _relative(input_path),
            "sha256": _sha256(input_path),
            "protocol_purpose": data["protocol"]["purpose"],
            "case_count": len(rows),
        },
        "model": model_metadata(learned_model),
        "cases": cases,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
