"""Independent quality-curve and descriptive-statistics calculations."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence


def area_under_log_curve(points: Sequence[tuple[int, float]], horizon: int) -> float | None:
    """Integrate a right-continuous incumbent curve over ``log(1 + resource)``.

    Quality is undefined before the first feasible checkpoint, so that interval is omitted. The
    incumbent at a checkpoint is carried forward until the next checkpoint or the supplied
    horizon. Duplicate resource positions retain the last quality observed there.
    """

    if horizon < 0:
        raise ValueError("curve horizon must be non-negative")
    if not points:
        return None
    collapsed: dict[int, float] = {}
    previous = -1
    for resource, quality in points:
        if resource < 0:
            raise ValueError("curve resource values must be non-negative")
        if resource < previous:
            raise ValueError("curve checkpoints must be ordered")
        if not math.isfinite(quality):
            raise ValueError("curve quality values must be finite")
        collapsed[resource] = quality
        previous = resource
    ordered = tuple(collapsed.items())
    if horizon < ordered[-1][0]:
        raise ValueError("curve horizon cannot precede the final checkpoint")
    area = 0.0
    for index, (resource, quality) in enumerate(ordered):
        stop = ordered[index + 1][0] if index + 1 < len(ordered) else horizon
        if stop > resource:
            area += quality * (math.log1p(stop) - math.log1p(resource))
    return area


def incumbent_quality_at(
    points: Sequence[tuple[int, float]], thresholds: Sequence[int]
) -> dict[int, float | None]:
    """Sample a continuing incumbent curve at fixed resource checkpoints."""

    previous = -1
    for resource, quality in points:
        if resource < 0 or resource < previous:
            raise ValueError("curve checkpoints must be ordered and non-negative")
        if not math.isfinite(quality):
            raise ValueError("curve quality values must be finite")
        previous = resource
    sampled: dict[int, float | None] = {}
    for threshold in thresholds:
        if threshold < 0:
            raise ValueError("checkpoint thresholds must be non-negative")
        available = [quality for resource, quality in points if resource <= threshold]
        sampled[threshold] = available[-1] if available else None
    return sampled


def relative_gain(current: float, baseline: float) -> float:
    """Return signed baseline-relative improvement on a higher-is-better quality scale."""

    denominator = abs(baseline)
    if denominator <= 1e-12:
        return current - baseline
    return (current - baseline) / denominator


def normalized_gap_closed(current: float, baseline: float, reference: float) -> float | None:
    """Return the fraction of an exact/best-known primary-quality gap closed."""

    denominator = reference - baseline
    if abs(denominator) <= 1e-12:
        return 1.0 if current >= reference - 1e-12 else None
    return (current - baseline) / denominator


def descriptive_summary(values: Iterable[float]) -> tuple[float | None, float | None, float | None]:
    """Return mean, sample variance, and a labeled normal-approximation 95% half-width."""

    finite = tuple(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None, None, None
    mean = statistics.fmean(finite)
    if len(finite) == 1:
        return mean, None, None
    variance = statistics.variance(finite)
    return mean, variance, 1.96 * math.sqrt(variance / len(finite))
