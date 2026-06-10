from __future__ import annotations

import re

from evalrag.judge.client import JudgeClient
from evalrag.judge.parsing import VerdictParseError, extract_last_json_block
from evalrag.judge.prompts import load_prompt
from evalrag.types import EvalCase, MetricResult

METRIC = "faithfulness"

# Splits on sentence-ending punctuation followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def decompose_claims(answer: str) -> list[str]:
    """Split an answer into atomic claims (deterministic).

    Sentence-level split is a defensible first pass: each sentence is treated
    as one checkable claim. Empty/whitespace fragments are dropped.
    """
    answer = answer.strip()
    if not answer:
        return []
    parts = _SENTENCE_SPLIT.split(answer)
    return [p.strip() for p in parts if p.strip()]


def parse_verdict(raw: str) -> list[bool]:
    """Extract per-claim supported booleans, ordered by 'index'."""
    data = extract_last_json_block(raw)
    try:
        claims = data["claims"]
        ordered = sorted(claims, key=lambda c: c["index"])
        return [bool(c["supported"]) for c in ordered]
    except (KeyError, TypeError) as e:
        raise VerdictParseError(f"malformed verdict block: {e}") from e


def _render_prompt(template: str, claims: list[str], case: EvalCase) -> str:
    claims_block = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    context_block = "\n\n".join(
        f"[{chunk.id}] {chunk.text}" for chunk in case.retrieved_chunks
    )
    return template.format(claims=claims_block, context=context_block)


def judge_claims(
    case: EvalCase,
    judge: JudgeClient,
    *,
    prompts_dir: str = "prompts",
) -> tuple[list[str], list[bool], str | None]:
    """Decompose the answer, ask the judge, and return (claims, verdicts, raw_output).

    Shared by score_faithfulness and the validation harness so the judge-calling
    path lives in one place. raw_output is the judge's verbatim response (None when
    there are no claims and thus no judge call). Raises VerdictParseError if the
    judge's verdict count doesn't match the claim count.
    """
    claims = decompose_claims(case.generated_answer)
    if not claims:
        return [], [], None

    version = judge.version_for(METRIC)
    template = load_prompt(METRIC, version, prompts_dir=prompts_dir)
    prompt = _render_prompt(template, claims, case)

    result = judge.judge(METRIC, prompt)
    verdicts = parse_verdict(result.raw_output)

    if len(verdicts) != len(claims):
        raise VerdictParseError(
            f"judge returned {len(verdicts)} verdicts for {len(claims)} claims"
        )
    return claims, verdicts, result.raw_output


def score_faithfulness(
    case: EvalCase,
    judge: JudgeClient,
    *,
    prompts_dir: str = "prompts",
) -> MetricResult:
    claims, verdicts, raw_output = judge_claims(case, judge, prompts_dir=prompts_dir)

    # Edge case: answer with no extractable claims.
    if not claims:
        return MetricResult(
            metric=METRIC,
            score=1.0,
            rationale="No atomic claims extracted from the answer; nothing to verify.",
            raw_judge_output=None,
        )

    supported = sum(verdicts)
    score = supported / len(claims)
    lines = [f"{'✓' if ok else '✗'} {claim}" for claim, ok in zip(claims, verdicts)]
    rationale = (
        f"{supported}/{len(claims)} claims grounded in retrieved context.\n"
        + "\n".join(lines)
    )
    return MetricResult(
        metric=METRIC,
        score=score,
        rationale=rationale,
        raw_judge_output=raw_output,
    )