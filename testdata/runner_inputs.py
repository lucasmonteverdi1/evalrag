from __future__ import annotations

from evalrag.runner.loop import EvalInput
from evalrag.types import Chunk

# Hand-built questions to run through the (fake) pipeline. The future generator/
# will emit this same EvalInput shape.
INPUTS: list[EvalInput] = [
    EvalInput(
        question="What is the capital of France?",
        expected_answer="Paris.",
        source_chunk_id="c1",
    ),
    EvalInput(
        question="What language is spoken in Brazil?",
        expected_answer="Portuguese.",
        source_chunk_id="c2",
    ),
    EvalInput(
        # No canned response below -> the fake adapter raises -> captured as a RunError.
        question="What is the meaning of life?",
    ),
]

# Canned (chunks, answer) the FakePipelineAdapter returns per question.
RESPONSES: dict[str, tuple[list[Chunk], str]] = {
    "What is the capital of France?": (
        [Chunk("c1", "Paris is the capital of France.")],
        "Paris is the capital of France.",
    ),
    "What language is spoken in Brazil?": (
        [Chunk("c2", "The official language of Brazil is Portuguese.")],
        "Portuguese is spoken in Brazil.",
    ),
}

# Questions the fake adapter should fail on (exercises collect-and-continue).
RAISES: dict[str, Exception] = {
    "What is the meaning of life?": RuntimeError("pipeline timed out"),
}
