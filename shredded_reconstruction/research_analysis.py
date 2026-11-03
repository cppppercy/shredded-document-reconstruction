"""Read-only summaries for the final independent confirmation experiment.

The functions operate only on stored per-case predictions.  They never train a
model or rerun a reconstruction, which keeps the research dashboard responsive
and prevents its charts from being mistaken for fresh interactive results.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


RULE_METHOD = "traditional_v0_2"
LEARNED_METHOD = "learned_v03_full"

DOCUMENT_TYPE_NAMES = {
    "report": "文字报告",
    "table": "表格文档",
    "mixed": "图文混排",
    "sparse": "稀疏文档",
}
CONDITION_NAMES_ZH = {
    "clean": "清晰完整",
    "no_assists": "无明显边框或纹理",
    "scan": "模糊及亮度变化",
    "edge_loss": "边缘缺失",
    "vertical_shift": "上下错位",
}


def load_confirmation(path: Path) -> list[dict[str, Any]]:
    """Load one immutable, identity-isolated experiment without mixing batches."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data.get("confirmation_cases", data.get("cases", ())))
    expected = data["protocol"].get("confirmation_cases", data["protocol"].get("cases"))
    if expected is None or len(rows) != int(expected):
        raise ValueError("experiment row count does not match its protocol")
    return rows


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, float | int]:
    if not rows:
        return {"samples": 0, "adjacency": 0.0, "exact": 0.0}
    return {
        "samples": len(rows),
        "adjacency": float(np.mean([row[f"{method}_adjacency"] for row in rows])),
        "exact": float(np.mean([bool(row[f"{method}_exact"]) for row in rows])),
    }


def comparison_table(rows: list[dict[str, Any]], *fields: str) -> list[dict[str, Any]]:
    """Summarize the two user-facing methods by one or more stored fields."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        rule = _method_summary(group, RULE_METHOD)
        learned = _method_summary(group, LEARNED_METHOD)
        item: dict[str, Any] = dict(zip(fields, key, strict=True))
        item.update({
            "samples": rule["samples"],
            "rule_adjacency": rule["adjacency"],
            "learned_adjacency": learned["adjacency"],
            "rule_exact": rule["exact"],
            "learned_exact": learned["exact"],
            "adjacency_delta_learned_minus_rule": learned["adjacency"] - rule["adjacency"],
        })
        output.append(item)
    return output


def family_bootstrap_interval(rows: list[dict[str, Any]], *, seed: int = 20261103, draws: int = 2000) -> dict[str, tuple[float, float]]:
    """95% bootstrap intervals after first averaging observations per document.

    Conditions from the same original page are correlated, so resampling rows
    would exaggerate the amount of independent evidence.  Here the unit is the
    original document identity (`family_id`), not the 120 observations.
    """
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family_id"])].append(row)
    family_rule = np.asarray([np.mean([row[f"{RULE_METHOD}_adjacency"] for row in group]) for group in by_family.values()])
    family_learned = np.asarray([np.mean([row[f"{LEARNED_METHOD}_adjacency"] for row in group]) for group in by_family.values()])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(family_rule), size=(draws, len(family_rule)))
    sampled_rule = family_rule[indices].mean(axis=1)
    sampled_learned = family_learned[indices].mean(axis=1)
    sampled_delta = sampled_learned - sampled_rule
    return {
        "rule": tuple(map(float, np.quantile(sampled_rule, (0.025, 0.975)))),
        "learned": tuple(map(float, np.quantile(sampled_learned, (0.025, 0.975)))),
        "delta": tuple(map(float, np.quantile(sampled_delta, (0.025, 0.975)))),
    }


def case_categories(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Classify real saved predictions into four useful comparison outcomes."""
    result = {"both_success": [], "rule_only": [], "learned_only": [], "both_fail": []}
    for row in rows:
        rule = bool(row[f"{RULE_METHOD}_exact"])
        learned = bool(row[f"{LEARNED_METHOD}_exact"])
        result["both_success" if rule and learned else "rule_only" if rule else "learned_only" if learned else "both_fail"].append(row)
    return result


def concise_conclusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return evidence-backed text fragments for the dashboard overview."""
    overall = comparison_table(rows)[0]
    by_condition = comparison_table(rows, "condition")
    strongest = max(by_condition, key=lambda row: abs(float(row["adjacency_delta_learned_minus_rule"])))
    interval = family_bootstrap_interval(rows)
    delta = float(overall["adjacency_delta_learned_minus_rule"])
    leader = "学习匹配" if delta > 0 else "规则匹配"
    return {
        "overall": overall,
        "leader": leader,
        "delta": delta,
        "strongest_condition": strongest,
        "interval": interval,
        "uncertain": interval["delta"][0] <= 0 <= interval["delta"][1],
    }
