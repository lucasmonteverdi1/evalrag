from __future__ import annotations

from evalrag.judge.llm_client import LLMClient
from evalrag.report.aggregate import CaseReport
from evalrag.runner.loop import RunResult
from evalrag.scorer.answer_relevance import score_answer_relevance
from evalrag.scorer.context_precision import score_context_precision
from evalrag.scorer.faithfulness import score_faithfulness
from evalrag.types import EvalCase, MetricResult

# The three MVP metrics, keyed by name. Each takes (case, judge) and returns a
# MetricResult. context_precision uses the judge only on its fallback path.
SCORERS = {
    "faithfulness": score_faithfulness,
    "answer_relevance": score_answer_relevance,
    "context_precision": score_context_precision,
}


def score_case(
    case: EvalCase,
    judge: LLMClient,
    *,
    metrics: list[str] | None = None,
    prompts_dir: str | None = None,
) -> list[MetricResult]:
    """Run the selected metrics over one EvalCase. Defaults to all three."""
    names = metrics if metrics is not None else list(SCORERS)
    results: list[MetricResult] = []
    for name in names:
        scorer = SCORERS[name]
        results.append(scorer(case, judge, prompts_dir=prompts_dir))
    return results


def score_run(
    run: RunResult,
    judge: LLMClient,
    *,
    metrics: list[str] | None = None,
    prompts_dir: str | None = None,
) -> list[CaseReport]:
    """Score every successful case in a RunResult, producing CaseReports."""
    return [
        CaseReport(case=case, results=score_case(case, judge, metrics=metrics, prompts_dir=prompts_dir))
        for case in run.cases
    ]
