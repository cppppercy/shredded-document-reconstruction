import numpy as np

from shredded_reconstruction.optimizer import solve_global_order


def test_global_optimizer_finds_known_path():
    count = 7
    costs = np.full((count, count), 9.0)
    np.fill_diagonal(costs, np.inf)
    expected = [3, 1, 6, 0, 5, 2, 4]
    for left, right in zip(expected, expected[1:]):
        costs[left, right] = 0.1
    start = np.full(count, 5.0)
    end = np.full(count, 5.0)
    start[expected[0]] = 0.0
    end[expected[-1]] = 0.0
    order, total = solve_global_order(costs, start, end)
    assert order == expected
    assert total < 1.0
