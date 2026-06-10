import pytest

from evalrag.config import ProviderConfig
from evalrag.judge.cache import ResponseCache
from evalrag.judge.client import JudgeClient
from evalrag.judge.parsing import VerdictParseError
from evalrag.judge.provider import FakeProvider
from evalrag.scorer.answer_relevance import parse_verdict, score_answer_relevance
from evalrag.types import Chunk, EvalCase
from testdata.answer_relevance_stubs import FAKE_RESPONSES, STUBS


class TestParseVerdict:
    def test_parses_score_and_reason(self):
        raw = 'reasoning...\n```json\n{"score": 0.85, "reason": "on topic"}\n```'
        score, reason = parse_verdict(raw)
        assert score == 0.85
        assert reason == "on topic"

    def test_reason_optional(self):
        score, reason = parse_verdict('```json\n{"score": 0.4}\n```')
        assert score == 0.4
        assert reason == ""

    def test_clamps_above_one(self):
        score, _ = parse_verdict('```json\n{"score": 1.7}\n```')
        assert score == 1.0

    def test_clamps_below_zero(self):
        score, _ = parse_verdict('```json\n{"score": -0.3}\n```')
        assert score == 0.0

    def test_uses_last_block(self):
        raw = '```json\n{"score": 0.1}\n```\nfinal:\n```json\n{"score": 0.9}\n```'
        score, _ = parse_verdict(raw)
        assert score == 0.9

    def test_no_block_raises(self):
        with pytest.raises(VerdictParseError, match="no JSON verdict block"):
            parse_verdict("no json here")

    def test_missing_score_raises(self):
        with pytest.raises(VerdictParseError, match="malformed"):
            parse_verdict('```json\n{"reason": "x"}\n```')

    def test_non_numeric_score_raises(self):
        with pytest.raises(VerdictParseError, match="malformed"):
            parse_verdict('```json\n{"score": "high"}\n```')


def make_judge(tmp_path, responses) -> JudgeClient:
    return JudgeClient(
        provider=FakeProvider(responses=responses, default='```json\n{"score": 0.0}\n```'),
        config=ProviderConfig(
            provider="fake",
            base_url="n/a",
            api_key_env="UNUSED",
            model="fake-model",
            temperature=0,
            max_tokens=1024,
        ),
        cache=ResponseCache(enabled=False, cache_dir=tmp_path),
        prompt_versions={"answer_relevance": "v1"},
    )


class TestScoreAnswerRelevance:
    def test_end_to_end(self, tmp_path):
        case = EvalCase(
            question="What is the capital of France?",
            generated_answer="The capital of France is Paris.",
            retrieved_chunks=(Chunk("c0", "n/a"),),
        )
        verdict = '```json\n{"score": 0.9, "reason": "directly answers"}\n```'
        judge = make_judge(tmp_path, {"The capital of France": verdict})
        result = score_answer_relevance(case, judge)
        assert result.metric == "answer_relevance"
        assert result.score == 0.9
        assert result.rationale == "directly answers"
        assert result.raw_judge_output == verdict

    def test_rationale_falls_back_when_no_reason(self, tmp_path):
        case = EvalCase(
            question="q",
            generated_answer="some answer",
            retrieved_chunks=(Chunk("c0", "n/a"),),
        )
        judge = make_judge(tmp_path, {"some answer": '```json\n{"score": 0.6}\n```'})
        result = score_answer_relevance(case, judge)
        assert "0.60" in result.rationale


class TestStubs:
    @pytest.mark.parametrize("stub", STUBS, ids=lambda s: s.case.question[:30])
    def test_each_stub_scores_to_its_label(self, tmp_path, stub):
        judge = make_judge(tmp_path, FAKE_RESPONSES)
        result = score_answer_relevance(stub.case, judge)
        assert result.score == pytest.approx(stub.label)

    def test_fake_keys_unique(self):
        assert len(FAKE_RESPONSES) == len(STUBS)
