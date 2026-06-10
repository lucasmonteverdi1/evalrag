from __future__ import annotations

import json
from dataclasses import dataclass

from evalrag.types import Chunk, EvalCase


def verdict_json(labels: list[bool]) -> str:
    """Build the canned per-chunk relevance verdict from ground-truth labels."""
    chunks = [{"index": i, "relevant": r} for i, r in enumerate(labels)]
    return f'```json\n{json.dumps({"chunks": chunks})}\n```'


@dataclass(frozen=True)
class PrecisionStub:
    """A context-precision case plus its expected score.

    `chunk_labels` is the per-chunk relevance ground truth (for the judge path);
    `expected_score` is the hand-computed (precision@k + MRR) / 2 for this case.
    """

    case: EvalCase
    chunk_labels: list[bool]
    expected_score: float


# --- Deterministic-path stubs (EvalCase carries source_chunk_id) ---
DETERMINISTIC_STUBS: list[PrecisionStub] = [
    PrecisionStub(
        # Relevant chunk ranked first: p@k = 1/2, MRR = 1.0 -> 0.75
        case=EvalCase(
            question="What is the capital of France?",
            generated_answer="Paris.",
            retrieved_chunks=(
                Chunk("c1", "Paris is the capital of France."),
                Chunk("c2", "The Amazon flows through Brazil."),
            ),
            source_chunk_id="c1",
        ),
        chunk_labels=[True, False],
        expected_score=0.75,
    ),
    PrecisionStub(
        # Relevant chunk ranked second: p@k = 1/2, MRR = 0.5 -> 0.5
        case=EvalCase(
            question="What language is spoken in Brazil?",
            generated_answer="Portuguese.",
            retrieved_chunks=(
                Chunk("c9", "The Eiffel Tower is in Paris."),
                Chunk("c2", "Portuguese is the official language of Brazil."),
            ),
            source_chunk_id="c2",
        ),
        chunk_labels=[False, True],
        expected_score=0.5,
    ),
]


# --- Judge-path stubs (no source_chunk_id; judge labels each chunk) ---
JUDGE_STUBS: list[PrecisionStub] = [
    PrecisionStub(
        # Both relevant: p@k = 1.0, MRR = 1.0 -> 1.0
        case=EvalCase(
            question="Tell me about Paris.",
            generated_answer="...",
            retrieved_chunks=(
                Chunk("c1", "Paris is the capital of France."),
                Chunk("c2", "The Seine river runs through Paris."),
            ),
        ),
        chunk_labels=[True, True],
        expected_score=1.0,
    ),
    PrecisionStub(
        # None relevant: p@k = 0.0, MRR = 0.0 -> 0.0
        case=EvalCase(
            question="What is the population of Tokyo?",
            generated_answer="...",
            retrieved_chunks=(
                Chunk("c1", "Bananas are a good source of potassium."),
                Chunk("c2", "The violin has four strings."),
            ),
        ),
        chunk_labels=[False, False],
        expected_score=0.0,
    ),
]


# FakeProvider responses for the judge path, keyed on a unique substring of each prompt
# (the first chunk's text appears verbatim in the rendered prompt).
JUDGE_FAKE_RESPONSES = {
    stub.case.retrieved_chunks[0].text: verdict_json(stub.chunk_labels)
    for stub in JUDGE_STUBS
}
