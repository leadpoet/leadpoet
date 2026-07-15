"""Read-only calibration metrics shared by tree monitoring and replay reports."""

from __future__ import annotations

import math
from typing import Sequence


MIN_HISTORICAL_PAIR_COUNT = 30
MIN_SPEARMAN_RHO = 0.20


def spearman_correlation(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    x_ranks = _ranks([pair[0] for pair in pairs])
    y_ranks = _ranks([pair[1] for pair in pairs])
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    numerator = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_ranks, y_ranks)
    )
    x_var = sum((x - x_mean) ** 2 for x in x_ranks)
    y_var = sum((y - y_mean) ** 2 for y in y_ranks)
    if x_var <= 0 or y_var <= 0:
        return 0.0
    return round(numerator / math.sqrt(x_var * y_var), 6)


def top_quartile_lift(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 4:
        return None
    ordered = sorted(pairs, key=lambda pair: pair[0], reverse=True)
    top_count = max(1, math.ceil(len(ordered) / 4))
    top_mean = sum(pair[1] for pair in ordered[:top_count]) / top_count
    overall_mean = sum(pair[1] for pair in ordered) / len(ordered)
    return round(top_mean - overall_mean, 6)


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for original_index, _value in ordered[index:end]:
            ranks[original_index] = average
        index = end
    return ranks
