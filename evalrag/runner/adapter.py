from __future__ import annotations

from typing import Protocol

from evalrag.types import Chunk


class PipelineAdapter(Protocol):
    def run(self, question: str) -> tuple[list[Chunk], str]:
        """Run the system-under-test for one question.

        Returns (retrieved_chunks, generated_answer). The tool treats the
        pipeline as a black box — it must not assume how retrieval/generation
        are implemented internally.
        """
        ...


class FakePipelineAdapter:
    """Deterministic PipelineAdapter test double. No network, no key.

    `responses` maps a question -> its canned (chunks, answer). Questions listed
    in `raises` make run() raise, to exercise the runner's error path. Unknown
    questions raise KeyError. Records call count.
    """

    def __init__(
        self,
        responses: dict[str, tuple[list[Chunk], str]],
        raises: dict[str, Exception] | None = None,
    ) -> None:
        self.responses = responses
        self.raises = raises or {}
        self.calls = 0

    def run(self, question: str) -> tuple[list[Chunk], str]:
        self.calls += 1
        if question in self.raises:
            raise self.raises[question]
        return self.responses[question]