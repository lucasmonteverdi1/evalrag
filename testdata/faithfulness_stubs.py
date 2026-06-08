import json

from evalrag.types import Chunk, EvalCase
from testdata.stubs import MetricStub


def verdict_json(labels: list[bool]) -> str:
    """Build the canned judge response (a JSON verdict block) from ground-truth labels.

    Generating it from labels guarantees the fake verdict can never drift out of
    sync with the claims that decompose_claims() produces.
    """
    claims = [{"index": i, "supported": s} for i, s in enumerate(labels)]
    return f'```json\n{json.dumps({"claims": claims})}\n```'


def _fake_key(answer: str) -> str:
    """A substring of `answer` that (a) appears verbatim in the rendered prompt and
    (b) is unique per stub. The LAST sentence satisfies both: each claim is rendered
    verbatim in the prompt, and our stubs' final sentences are all distinct (unlike
    their first sentences, where two stubs share "Paris is the capital of France").
    """
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    return sentences[-1]


STUBS: list[MetricStub[list[bool]]] = [
    MetricStub(
        # Fully grounded: every claim is backed by the context (score 1.0).
        case=EvalCase(
            question="What is the capital of France and what river runs through it?",
            generated_answer="Paris is the capital of France. The Seine river runs through Paris.",
            retrieved_chunks=(
                Chunk("c1", "Paris is the capital of France."),
                Chunk("c2", "The Seine is a river that flows through Paris."),
            ),
        ),
        label=[True, True],
    ),
    MetricStub(
        # Partially grounded: one true claim, one hallucinated number (score 0.5).
        case=EvalCase(
            question="Tell me about Paris.",
            generated_answer="Paris is the capital of France. Paris has exactly 12 million inhabitants.",
            retrieved_chunks=(
                Chunk("c1", "Paris is the capital of France."),
            ),
        ),
        label=[True, False],
    ),
    MetricStub(
        # Fully hallucinated: context says nothing relevant (score 0.0).
        case=EvalCase(
            question="What is the population of Tokyo?",
            generated_answer="Tokyo has 5 billion people. Tokyo is located in Brazil.",
            retrieved_chunks=(
                Chunk("c1", "Tokyo is the capital of Japan."),
            ),
        ),
        label=[False, False],
    ),
    MetricStub(
        # Single claim, grounded (score 1.0).
        case=EvalCase(
            question="What language is spoken in Brazil?",
            generated_answer="Portuguese is the official language of Brazil.",
            retrieved_chunks=(
                Chunk("c1", "The official language of Brazil is Portuguese."),
            ),
        ),
        label=[True],
    ),
    MetricStub(
        # Three claims, mixed (score ~0.67).
        case=EvalCase(
            question="Tell me about the sun.",
            generated_answer="The sun is a star. The sun is made mostly of hydrogen. The sun is cold.",
            retrieved_chunks=(
                Chunk("c1", "The Sun is a star at the center of the Solar System."),
                Chunk("c2", "The Sun is composed primarily of hydrogen and helium."),
            ),
        ),
        label=[True, True, False],
    ),
]


# Key each canned verdict on a substring that appears verbatim in the rendered prompt
# AND is unique per stub (see _fake_key), so FakeProvider returns the right verdict
# with no key collisions.
FAKE_RESPONSES = {
    _fake_key(stub.case.generated_answer): verdict_json(stub.label) for stub in STUBS
}