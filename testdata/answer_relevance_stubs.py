from __future__ import annotations

import json

from evalrag.types import Chunk, EvalCase
from testdata.stubs import MetricStub


def verdict_json(score: float, reason: str = "stub verdict") -> str:
    """Build the canned judge response (a JSON verdict block) from a target score."""
    return f'```json\n{json.dumps({"score": score, "reason": reason})}\n```'


# Each stub: an EvalCase + the hand-assigned ground-truth relevance score (0.0–1.0).
# retrieved_chunks are irrelevant to this metric (it reads question + answer only),
# but EvalCase requires them, so we pass a minimal placeholder.
_PLACEHOLDER = (Chunk("c0", "n/a"),)

STUBS: list[MetricStub[float]] = [
    MetricStub(
        # Fully on-topic and complete.
        case=EvalCase(
            question="What is the capital of France?",
            generated_answer="The capital of France is Paris.",
            retrieved_chunks=_PLACEHOLDER,
        ),
        label=1.0,
    ),
    MetricStub(
        # On-topic but incomplete / vague.
        case=EvalCase(
            question="What are the three primary colors?",
            generated_answer="Red is one of them.",
            retrieved_chunks=_PLACEHOLDER,
        ),
        label=0.5,
    ),
    MetricStub(
        # Confident but completely off-topic.
        case=EvalCase(
            question="How does photosynthesis work?",
            generated_answer="The stock market rose two percent on Tuesday.",
            retrieved_chunks=_PLACEHOLDER,
        ),
        label=0.0,
    ),
]


# Key each canned verdict on a unique substring of the prompt. The answer text appears
# verbatim in the rendered prompt and is unique per stub.
FAKE_RESPONSES = {
    stub.case.generated_answer: verdict_json(stub.label) for stub in STUBS
}
