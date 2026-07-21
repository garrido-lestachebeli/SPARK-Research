"""Pure-Python descriptive statistics for training telemetry."""

from __future__ import annotations

import math
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sample_std(values) / math.sqrt(len(values))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean_ci(values: list[float], *, z: float = 1.96) -> dict[str, float]:
    average = mean(values)
    half_width = z * stderr(values)
    return {
        "mean": average,
        "std": sample_std(values),
        "stderr": stderr(values),
        "ci_low": average - half_width,
        "ci_high": average + half_width,
        "ci_half_width": half_width,
        "n": float(len(values)),
        "median": median(values),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "sum": sum(values) if values else 0.0,
    }


def wilson_ci(successes: int, trials: int, *, z: float = 1.96) -> dict[str, float]:
    if trials <= 0:
        return {
            "rate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n": 0.0,
            "successes": 0.0,
        }
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = proportion + z**2 / (2 * trials)
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z**2 / (4 * trials)) / trials
    )
    return {
        "rate": proportion,
        "ci_low": max(0.0, (center - margin) / denominator),
        "ci_high": min(1.0, (center + margin) / denominator),
        "n": float(trials),
        "successes": float(successes),
    }


def summarize_outcomes(counts: dict[str, int]) -> dict[str, Any]:
    wins = int(counts.get("win", 0))
    losses = int(counts.get("loss", 0))
    draws = int(counts.get("draw", 0))
    total = wins + losses + draws
    decisive = wins + losses
    return {
        "win": wins,
        "loss": losses,
        "draw": draws,
        "total": total,
        "win_rate": wilson_ci(wins, total),
        "loss_rate": wilson_ci(losses, total),
        "draw_rate": wilson_ci(draws, total),
        "win_rate_excluding_draws": wilson_ci(wins, decisive),
    }
