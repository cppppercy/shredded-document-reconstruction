"""Exact global strip ordering with Held--Karp dynamic programming."""

from __future__ import annotations

import numpy as np


def solve_global_order(
    transition_costs: np.ndarray,
    start_costs: np.ndarray | None = None,
    end_costs: np.ndarray | None = None,
) -> tuple[list[int], float]:
    """Return the minimum-cost Hamiltonian path through all strip positions."""
    costs = np.asarray(transition_costs, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError("transition_costs must be a square matrix")
    count = costs.shape[0]
    if not 2 <= count <= 16:
        raise ValueError("exact optimization supports 2 to 16 strips")
    start = np.zeros(count) if start_costs is None else np.asarray(start_costs, dtype=np.float64)
    end = np.zeros(count) if end_costs is None else np.asarray(end_costs, dtype=np.float64)
    if start.shape != (count,) or end.shape != (count,):
        raise ValueError("endpoint costs must have one value per strip")
    if not np.all(np.isfinite(costs[np.eye(count, dtype=bool) == 0])):
        raise ValueError("off-diagonal transition costs must be finite")

    state_count = 1 << count
    dp = np.full((state_count, count), np.inf, dtype=np.float64)
    parent = np.full((state_count, count), -1, dtype=np.int16)
    for last in range(count):
        dp[1 << last, last] = start[last]

    for mask in range(1, state_count):
        for last in range(count):
            if not mask & (1 << last):
                continue
            previous_mask = mask ^ (1 << last)
            if previous_mask == 0:
                continue
            candidates = [
                (dp[previous_mask, previous] + costs[previous, last], previous)
                for previous in range(count)
                if previous_mask & (1 << previous)
            ]
            value, previous = min(candidates)
            dp[mask, last] = value
            parent[mask, last] = previous

    full = state_count - 1
    last = int(np.argmin(dp[full] + end))
    best_cost = float(dp[full, last] + end[last])
    order = []
    mask = full
    while last >= 0:
        order.append(last)
        previous = int(parent[mask, last])
        mask ^= 1 << last
        last = previous
    order.reverse()
    if len(order) != count:
        raise RuntimeError("failed to reconstruct the optimized path")
    return order, best_cost
