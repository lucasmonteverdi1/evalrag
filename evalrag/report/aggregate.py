from __future__ import annotations

from dataclasses import dataclass

from evalrag.types import EvalCase, MetricResult


@dataclass(frozen=True)
class CaseReport:
    """One evaluated case: the EvalCase plus the MetricResults scored for it."""

    case: EvalCase
    results: list[MetricResult]


@dataclass(frozen=True)
class Aggregate:
    """Aggregated scores across all cases.

    per_metric maps metric name -> mean score over the cases that have it. This is
    the gating-relevant value: the CLI compares EACH per_metric entry against its
    own threshold to decide pass/fail.

    overall is the mean of the per-metric means — INFORMATIONAL ONLY (a one-glance
    summary for the report). It is NOT used for threshold gating. None if there are
    no metrics.
    """

    per_metric: dict[str, float]
    overall: float | None
    n_cases: int


def aggregate(case_reports: list[CaseReport]) -> Aggregate:
    """Compute mean score per metric and an overall mean across all cases.

    Pure calculation — knows nothing about thresholds; gating lives in the CLI.
    """
    # Collect every score seen for each metric name.
    scores_by_metric: dict[str, list[float]] = {}
    for cr in case_reports:
        for result in cr.results:
            scores_by_metric.setdefault(result.metric, []).append(result.score)

    per_metric = {
        metric: sum(scores) / len(scores)
        for metric, scores in scores_by_metric.items()
    }
    # Macro-average: each metric weighs equally regardless of case count.
    overall = sum(per_metric.values()) / len(per_metric) if per_metric else None

    return Aggregate(
        per_metric=per_metric,
        overall=overall,
        n_cases=len(case_reports),
    )