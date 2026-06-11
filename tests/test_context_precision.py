import pytest

from evalrag.config import ProviderConfig
from evalrag.judge.cache import ResponseCache
from evalrag.judge.llm_client import LLMClient
from evalrag.judge.parsing import VerdictParseError
from evalrag.judge.provider import FakeProvider
from evalrag.scorer.context_precision import (
    mrr,
    precision_at_k,
    score_context_precision,
)
from evalrag.types import Chunk, EvalCase
from testdata.context_precision_stubs import (
    DETERMINISTIC_STUBS,
    JUDGE_FAKE_RESPONSES,
    JUDGE_STUBS,
)


class TestPrecisionAtK:
    def test_fraction_relevant(self):
        assert precision_at_k(["c1", "c2", "c3"], {"c1", "c2"}) == pytest.approx(2 / 3)

    def test_all_relevant(self):
        assert precision_at_k(["c1", "c2"], {"c1", "c2"}) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["c1", "c2"], {"c9"}) == 0.0

    def test_empty_retrieval(self):
        assert precision_at_k([], {"c1"}) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["c1", "c2"], {"c1"}) == 1.0

    def test_second_position(self):
        assert mrr(["c3", "c1"], {"c1"}) == 0.5

    def test_fourth_position(self):
        assert mrr(["a", "b", "c", "c1"], {"c1"}) == 0.25

    def test_none_relevant(self):
        assert mrr(["a", "b"], {"c1"}) == 0.0

    def test_uses_first_relevant_only(self):
        # Two relevant; MRR keys on the earliest (position 1).
        assert mrr(["c1", "c2"], {"c1", "c2"}) == 1.0


class TestDeterministicPath:
    @pytest.mark.parametrize("stub", DETERMINISTIC_STUBS, ids=lambda s: s.case.question[:25])
    def test_scores_match_expected(self, stub):
        # No judge passed: deterministic path must work without an LLM.
        result = score_context_precision(stub.case)
        assert result.metric == "context_precision"
        assert result.score == pytest.approx(stub.expected_score)
        assert result.raw_judge_output is None
        assert "deterministic" in result.rationale


def make_judge(tmp_path, responses) -> LLMClient:
    return LLMClient(
        provider=FakeProvider(
            responses=responses, default='```json\n{"chunks": []}\n```'
        ),
        config=ProviderConfig(
            provider="fake",
            base_url="n/a",
            api_key_env="UNUSED",
            model="fake-model",
            temperature=0,
            max_tokens=1024,
        ),
        cache=ResponseCache(enabled=False, cache_dir=tmp_path),
        prompt_versions={"context_precision": "v1"},
    )


class TestJudgePath:
    @pytest.mark.parametrize("stub", JUDGE_STUBS, ids=lambda s: s.case.question[:25])
    def test_scores_match_expected(self, tmp_path, stub):
        judge = make_judge(tmp_path, JUDGE_FAKE_RESPONSES)
        result = score_context_precision(stub.case, judge)
        assert result.score == pytest.approx(stub.expected_score)
        assert result.raw_judge_output is not None  # raw captured on judge path
        assert "judge per-chunk" in result.rationale

    def test_missing_judge_raises(self):
        # A case without source_chunk_id and no judge -> clear error.
        case = EvalCase(
            question="q",
            generated_answer="a",
            retrieved_chunks=(Chunk("c1", "x"),),
        )
        with pytest.raises(ValueError, match="needs a judge"):
            score_context_precision(case, judge=None)

    def test_verdict_count_mismatch_raises(self, tmp_path):
        case = EvalCase(
            question="q",
            generated_answer="a",
            retrieved_chunks=(Chunk("c1", "x"), Chunk("c2", "y")),
        )
        # Judge returns 1 verdict for 2 chunks.
        bad = '```json\n{"chunks": [{"index": 0, "relevant": true}]}\n```'
        judge = make_judge(tmp_path, {"x": bad})
        with pytest.raises(VerdictParseError, match="1 verdicts for 2 chunks"):
            score_context_precision(case, judge)
