"""Evaluate V0.1 on fixed held-out documents and save every result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shredded_reconstruction.evaluation import run_case
from shredded_reconstruction.learned import train_compatibility_model
from shredded_reconstruction.visualize import comparison_figure, evaluation_grid, shuffled_montage

EVALUATION_CASES = (
    (3001, "report", 8, 41),
    (3002, "table", 9, 42),
    (3003, "mixed", 10, 43),
    (3004, "sparse", 10, 44),
    (3005, "report", 11, 45),
    (3006, "table", 12, 46),
    (3007, "mixed", 12, 47),
    (3008, "sparse", 12, 48),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = train_compatibility_model()
    records = []
    thumbnails = []
    captions = []
    for case_index, (seed, variant, strip_count, shuffle_seed) in enumerate(EVALUATION_CASES, 1):
        result = run_case(
            seed=seed,
            variant=variant,
            strip_count=strip_count,
            shuffle_seed=shuffle_seed,
            model=model,
        )
        shuffled = shuffled_montage(result.shredded.strips)
        figure = comparison_figure(
            result.original,
            shuffled,
            result.baseline_reconstruction,
            result.learned_reconstruction,
            result.baseline_metrics,
            result.learned_metrics,
        )
        name = f"case-{case_index:02d}-{variant}-{strip_count}-strips.png"
        figure.save(args.output_dir / name, optimize=True)
        record = {
            "case": case_index,
            "seed": seed,
            "variant": variant,
            "strips": strip_count,
            "shuffle_seed": shuffle_seed,
            "baseline_adjacency": result.baseline_metrics["adjacency_accuracy"],
            "baseline_position": result.baseline_metrics["position_accuracy"],
            "baseline_exact": result.baseline_metrics["exact_match"],
            "baseline_order": result.baseline_metrics["predicted_original_order"],
            "learned_adjacency": result.learned_metrics["adjacency_accuracy"],
            "learned_position": result.learned_metrics["position_accuracy"],
            "learned_exact": result.learned_metrics["exact_match"],
            "learned_order": result.learned_metrics["predicted_original_order"],
            "image": name,
        }
        records.append(record)
        thumbnails.append(result.learned_reconstruction)
        captions.append(
            f"#{case_index} {variant}/{strip_count} · B {record['baseline_adjacency']:.0%} · ML {record['learned_adjacency']:.0%}"
        )
        print(json.dumps(record))

    summary = {
        "evaluation_documents": len(records),
        "training_document_ids": list(model.training_document_ids),
        "training_and_evaluation_documents_are_disjoint": all(
            str(record["seed"]) not in " ".join(model.training_document_ids) for record in records
        ),
        "baseline_mean_adjacency": sum(r["baseline_adjacency"] for r in records) / len(records),
        "learned_mean_adjacency": sum(r["learned_adjacency"] for r in records) / len(records),
        "baseline_exact_cases": sum(r["baseline_exact"] for r in records),
        "learned_exact_cases": sum(r["learned_exact"] for r in records),
        "cases": records,
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for key in records[0] if not key.endswith("_order")])
        writer.writeheader()
        writer.writerows({key: value for key, value in record.items() if not key.endswith("_order")} for record in records)
    evaluation_grid(thumbnails, captions).save(args.output_dir / "evaluation-grid.png", optimize=True)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"cases", "training_document_ids"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
