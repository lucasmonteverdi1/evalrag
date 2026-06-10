from __future__ import annotations

from dataclasses import dataclass

from evalrag.runner.adapter import PipelineAdapter
from evalrag.types import EvalCase


@dataclass(frozen=True)
class EvalInput:
    """One question to run through the pipeline, with optional ground truth.

    The future generator/ will emit these; for now they're hand-built in testdata.
    """

    question: str
    expected_answer: str | None = None
    source_chunk_id: str | None = None


@dataclass(frozen=True)
class RunError:
    """A per-question pipeline failure, captured so the run can continue."""

    question: str
    error: str


@dataclass(frozen=True)
class RunResult:
    """Output of a run: successful cases plus any captured failures."""

    cases: list[EvalCase]
    errors: list[RunError]


def run_pipeline(inputs: list[EvalInput], adapter: PipelineAdapter) -> RunResult:
    """Drive the pipeline over each input, building EvalCases.

    Collect-and-continue: a per-question exception is captured as a RunError and
    the loop moves on, so one bad question never aborts the whole run.
    """
    cases: list[EvalCase] = []
    errors: list[RunError] = []

    for inp in inputs:
        try:
            chunks, answer = adapter.run(inp.question)
        except Exception as e:  # noqa: BLE001 — intentional: isolate per-question failures
            errors.append(RunError(question=inp.question, error=str(e)))
            continue

        cases.append(
            EvalCase(
                question=inp.question,
                generated_answer=answer,
                retrieved_chunks=chunks,  # EvalCase.__post_init__ coerces list -> tuple
                expected_answer=inp.expected_answer,
                source_chunk_id=inp.source_chunk_id,
            )
        )

    return RunResult(cases=cases, errors=errors)