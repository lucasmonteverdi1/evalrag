from __future__ import annotations

from evalrag.judge.llm_client import LLMClient
from evalrag.judge.parsing import VerdictParseError, extract_last_json_block
from evalrag.judge.prompts import load_prompt
from evalrag.types import EvalCase, MetricResult

METRIC = "answer_relevance"


def parse_verdict(raw: str) -> tuple[float, str]:
    """Extract (score, reason) from the judge's raw output.

    Expects the LAST fenced JSON block: {"score": 0.0-1.0, "reason": "..."}.
    Clamps the score to [0, 1]. Raises VerdictParseError if malformed.
    """
    data = extract_last_json_block(raw)
    try:
        score = float(data["score"])
        reason = str(data.get("reason", ""))
    except (KeyError, TypeError, ValueError) as e:
        raise VerdictParseError(f"malformed verdict block: {e}") from e
    score = max(0.0, min(1.0, score))  # clamp to [0, 1]
    return score, reason


def _render_prompt(template: str, case: EvalCase) -> str:
    return template.format(question=case.question, answer=case.generated_answer)


def score_answer_relevance(
    case: EvalCase,
    judge: LLMClient,
    *,
    prompts_dir: str | None = None,
) -> MetricResult:
    version = judge.version_for(METRIC)
    template = load_prompt(METRIC, version, prompts_dir=prompts_dir)
    prompt = _render_prompt(template, case)

    result = judge.judge(METRIC, prompt)
    score, reason = parse_verdict(result.raw_output)

    rationale = reason or f"Relevance score: {score:.2f}"
    return MetricResult(
        metric=METRIC,
        score=score,
        rationale=rationale,
        raw_judge_output=result.raw_output,
    )