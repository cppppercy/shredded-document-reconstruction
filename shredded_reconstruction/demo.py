"""Command-line end-to-end V0.3 demonstration with explicit method names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conditions import CONDITION_NAMES, observation_condition
from .document import DOCUMENT_VARIANTS
from .evaluation import run_case
from .learned import train_compatibility_model, train_v03_compatibility_model
from .splits import TRAIN_DOCUMENTS
from .v03_splits import V03_TRAIN_DOCUMENTS
from .visualize import comparison_figure, shuffled_montage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct a synthetic vertically shredded document")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--variant", choices=DOCUMENT_VARIANTS, default="mixed")
    parser.add_argument("--strips", type=int, choices=range(8, 13), default=10)
    parser.add_argument("--shuffle-seed", type=int, default=73)
    parser.add_argument("--condition", choices=CONDITION_NAMES, default="clean")
    parser.add_argument(
        "--method",
        choices=("traditional-v0.1", "traditional-v0.2", "learned-v0.1", "learned-v0.2", "learned-v0.3"),
        default="learned-v0.3",
        help="The label names the actual compatibility implementation used.",
    )
    return parser


def _model(method: str):
    if method.startswith("traditional-"):
        return None
    if method == "learned-v0.1":
        return train_compatibility_model()
    if method == "learned-v0.2":
        return train_compatibility_model(TRAIN_DOCUMENTS)
    return train_v03_compatibility_model(
        V03_TRAIN_DOCUMENTS,
        augmentations=tuple(observation_condition(name) for name in CONDITION_NAMES),
        endpoint_multiplier=0.0,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = _model(args.method)
    result = run_case(
        seed=args.seed, variant=args.variant, strip_count=args.strips,
        shuffle_seed=args.shuffle_seed, model=model, condition_name=args.condition,
        baseline_max_vertical_shift=0 if args.method == "traditional-v0.1" else 3,
    )
    shuffled = shuffled_montage(result.shredded.strips)
    result.original.save(args.output_dir / "original.png")
    shuffled.save(args.output_dir / "shuffled.png")
    result.baseline_reconstruction.save(args.output_dir / "baseline-reconstruction.png")
    if result.learned_reconstruction is not None:
        result.learned_reconstruction.save(args.output_dir / "selected-reconstruction.png")
    comparison = comparison_figure(
        result.original, shuffled, result.baseline_reconstruction, result.learned_reconstruction,
        result.baseline_metrics, result.learned_metrics,
    )
    comparison.save(args.output_dir / "comparison.png")
    selected_metrics = result.baseline_metrics if args.method.startswith("traditional-") else result.learned_metrics
    metrics = {
        "case": {"seed": args.seed, "variant": args.variant, "strips": args.strips,
                 "shuffle_seed": args.shuffle_seed, "condition": args.condition},
        "selected_method": args.method,
        "shuffled_original_indices": list(result.shredded.original_indices),
        "traditional_control_metrics": result.baseline_metrics,
        "selected_method_metrics": selected_metrics,
        "training_documents": list(model.training_document_ids) if model else [],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved demo to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
