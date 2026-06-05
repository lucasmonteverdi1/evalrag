from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A retrieved document chunk."""

    id: str
    text: str


@dataclass(frozen=True)
class EvalCase:
    """One evaluation unit: a question run through the pipeline.

    Fields consumed per metric:
      - faithfulness:      generated_answer + retrieved_chunks
      - answer_relevance:  generated_answer + question
      - context_precision: retrieved_chunks + source_chunk_id (or question as fallback)
      - answer_correctness (optional): generated_answer + expected_answer
    """

    question: str
    generated_answer: str
    retrieved_chunks: tuple[Chunk, ...]  # frozen — use tuple, not list
    expected_answer: str | None = None
    source_chunk_id: str | None = None

    def __post_init__(self) -> None:
        # Coerce a plain list to tuple so callers don't have to care.
        if isinstance(self.retrieved_chunks, list):
            object.__setattr__(self, "retrieved_chunks", tuple(self.retrieved_chunks))


@dataclass(frozen=True)
class MetricResult:
    """Score produced by one metric for one EvalCase.

    raw_judge_output is stored verbatim — the full LLM response — for
    reproducibility audits and prompt debugging.
    """

    metric: str
    score: float  # normalized 0.0–1.0
    rationale: str
    raw_judge_output: str | None = None
