from __future__ import annotations

from dataclasses import dataclass

from evalrag.report.aggregate import Aggregate


@dataclass(frozen=True)
class MetricGate:
    """Pass/fail outcome for one metric against its threshold."""

    metric: str
    score: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    """Overall gating outcome. `passed` is True only if every gated metric passed."""

    gates: list[MetricGate]
    passed: bool


def evaluate_thresholds(agg: Aggregate, thresholds: dict[str, float]) -> GateResult:
    """Compare each metric's mean score against its threshold.

    A metric is gated only if it has BOTH a score (it was evaluated) and a
    non-null threshold. The run passes iff every gated metric is at or above its
    threshold. `overall` is intentionally ignored — gating is per-metric.
    """
    gates: list[MetricGate] = []
    for metric, score in sorted(agg.per_metric.items()):
        threshold = thresholds.get(metric)
        if threshold is None:
            continue  # metric not gated (no threshold configured)
        gates.append(
            MetricGate(
                metric=metric,
                score=score,
                threshold=float(threshold),
                passed=score >= float(threshold),
            )
        )
    passed = all(g.passed for g in gates)
    return GateResult(gates=gates, passed=passed)
