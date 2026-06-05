import pytest
from dataclasses import FrozenInstanceError

from evalrag.types import Chunk, EvalCase, MetricResult


class TestChunk:
    def test_basic(self):
        c = Chunk(id="c1", text="hello world")
        assert c.id == "c1"
        assert c.text == "hello world"

    def test_frozen(self):
        c = Chunk(id="c1", text="hello")
        with pytest.raises(FrozenInstanceError):
            c.id = "other"  # type: ignore[misc]


class TestEvalCase:
    def test_basic_with_tuple(self):
        chunks = (Chunk("c1", "ctx"), Chunk("c2", "more ctx"))
        ec = EvalCase(
            question="What is X?",
            generated_answer="X is Y.",
            retrieved_chunks=chunks,
        )
        assert ec.question == "What is X?"
        assert len(ec.retrieved_chunks) == 2
        assert ec.expected_answer is None
        assert ec.source_chunk_id is None

    def test_list_coerced_to_tuple(self):
        # Callers may pass a list; __post_init__ coerces it.
        chunks = [Chunk("c1", "ctx")]
        ec = EvalCase(
            question="Q?",
            generated_answer="A.",
            retrieved_chunks=chunks,  # type: ignore[arg-type]
        )
        assert isinstance(ec.retrieved_chunks, tuple)

    def test_optional_fields(self):
        ec = EvalCase(
            question="Q?",
            generated_answer="A.",
            retrieved_chunks=(),
            expected_answer="Expected.",
            source_chunk_id="c42",
        )
        assert ec.expected_answer == "Expected."
        assert ec.source_chunk_id == "c42"

    def test_frozen(self):
        ec = EvalCase(question="Q?", generated_answer="A.", retrieved_chunks=())
        with pytest.raises(FrozenInstanceError):
            ec.question = "other"  # type: ignore[misc]


class TestMetricResult:
    def test_basic(self):
        mr = MetricResult(metric="faithfulness", score=0.85, rationale="4 of 5 claims supported.")
        assert mr.metric == "faithfulness"
        assert mr.score == 0.85
        assert mr.raw_judge_output is None

    def test_with_raw_output(self):
        mr = MetricResult(
            metric="faithfulness",
            score=1.0,
            rationale="All claims grounded.",
            raw_judge_output='{"verdict": "SUPPORTED", "claims": [...]}',
        )
        assert mr.raw_judge_output is not None

    def test_frozen(self):
        mr = MetricResult(metric="faithfulness", score=0.5, rationale="Half.")
        with pytest.raises(FrozenInstanceError):
            mr.score = 0.9  # type: ignore[misc]

    def test_score_range(self):
        # Score is a float; no clamping in the dataclass itself — scorers enforce this.
        mr = MetricResult(metric="m", score=0.0, rationale="r")
        assert 0.0 <= mr.score <= 1.0
