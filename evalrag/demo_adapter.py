from __future__ import annotations

from evalrag.types import Chunk

# A tiny in-memory "knowledge base" so the demo adapter can answer a few questions
# without any real retrieval/generation stack. Load it via --adapter:
#   evalrag.demo_adapter:demo_adapter
_KB: dict[str, tuple[list[Chunk], str]] = {
    "What is the capital of France?": (
        [Chunk("c1", "Paris is the capital of France.")],
        "Paris is the capital of France.",
    ),
    "What language is spoken in Brazil?": (
        [Chunk("c2", "The official language of Brazil is Portuguese.")],
        "Portuguese is spoken in Brazil.",
    ),
}


class DemoPipelineAdapter:
    """A trivial PipelineAdapter for trying evalrag end-to-end with no real pipeline.

    Returns canned (chunks, answer) for known questions; for unknown questions it
    returns no chunks and a generic non-answer (so metrics score it low).
    """

    def run(self, question: str) -> tuple[list[Chunk], str]:
        if question in _KB:
            return _KB[question]
        return [], "I don't know."


# Module-level instance for the dotted-path loader.
demo_adapter = DemoPipelineAdapter()
