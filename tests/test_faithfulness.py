import pytest

from evalrag.judge.cache import ResponseCache
from evalrag.judge.llm_client import LLMClient
from evalrag.judge.provider import FakeProvider
from evalrag.config import ProviderConfig
from evalrag.scorer.faithfulness import (
    VerdictParseError,
    decompose_claims,
    parse_verdict,
    score_faithfulness,
)
from evalrag.types import Chunk, EvalCase
from testdata.faithfulness_stubs import FAKE_RESPONSES, STUBS


# --------------------------------------------------------------------------- #
# decompose_claims                                                            #
# --------------------------------------------------------------------------- #
class TestDecomposeClaims:
    def test_splits_on_sentence_boundaries(self):
        claims = decompose_claims("Paris is the capital. The Seine runs through it.")
        assert claims == ["Paris is the capital.", "The Seine runs through it."]

    def test_single_sentence(self):
        assert decompose_claims("Only one claim here.") == ["Only one claim here."]

    def test_handles_question_and_exclamation(self):
        claims = decompose_claims("Is it true? Yes it is! Definitely.")
        assert claims == ["Is it true?", "Yes it is!", "Definitely."]

    def test_empty_string_returns_empty_list(self):
        assert decompose_claims("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert decompose_claims("   \n  ") == []

    def test_strips_and_drops_blank_fragments(self):
        # Trailing period produces no empty trailing claim.
        assert decompose_claims("A claim.") == ["A claim."]


# --------------------------------------------------------------------------- #
# parse_verdict                                                               #
# --------------------------------------------------------------------------- #
class TestParseVerdict:
    def test_parses_well_formed_block(self):
        raw = 'reasoning...\n```json\n{"claims": [{"index": 0, "supported": true}, {"index": 1, "supported": false}]}\n```'
        assert parse_verdict(raw) == [True, False]

    def test_orders_by_index(self):
        raw = '```json\n{"claims": [{"index": 1, "supported": true}, {"index": 0, "supported": false}]}\n```'
        assert parse_verdict(raw) == [False, True]  # reordered to index 0,1

    def test_uses_last_json_block(self):
        # An earlier JSON-looking block (e.g. from chain-of-thought) is ignored.
        raw = (
            '```json\n{"claims": [{"index": 0, "supported": false}]}\n```\n'
            'final answer:\n'
            '```json\n{"claims": [{"index": 0, "supported": true}]}\n```'
        )
        assert parse_verdict(raw) == [True]

    def test_no_block_raises(self):
        with pytest.raises(VerdictParseError, match="no JSON verdict block"):
            parse_verdict("the model rambled without any json")

    def test_malformed_json_raises(self):
        raw = '```json\n{"claims": [{"index": 0, "supported": tru }]}\n```'
        with pytest.raises(VerdictParseError, match="malformed"):
            parse_verdict(raw)

    def test_missing_supported_key_raises(self):
        raw = '```json\n{"claims": [{"index": 0}]}\n```'
        with pytest.raises(VerdictParseError, match="malformed"):
            parse_verdict(raw)


# --------------------------------------------------------------------------- #
# score_faithfulness (end-to-end with FakeProvider)                           #
# --------------------------------------------------------------------------- #
def make_judge(tmp_path, responses, *, default="") -> LLMClient:
    return LLMClient(
        provider=FakeProvider(responses=responses, default=default),
        config=ProviderConfig(
            provider="fake",
            base_url="n/a",
            api_key_env="UNUSED",
            model="fake-model",
            temperature=0,
            max_tokens=1024,
        ),
        cache=ResponseCache(enabled=False, cache_dir=tmp_path),
        prompt_versions={"faithfulness": "v1"},
    )


class TestScoreFaithfulness:
    def test_exact_fraction(self, tmp_path):
        case = EvalCase(
            question="q",
            generated_answer="Claim A is here. Claim B is here. Claim C is here.",
            retrieved_chunks=(Chunk("c1", "context"),),
        )
        verdict = '```json\n{"claims": [{"index": 0, "supported": true}, {"index": 1, "supported": true}, {"index": 2, "supported": false}]}\n```'
        judge = make_judge(tmp_path, {"Claim A": verdict})
        result = score_faithfulness(case, judge)
        assert result.metric == "faithfulness"
        assert result.score == pytest.approx(2 / 3)
        assert result.raw_judge_output == verdict

    def test_empty_claims_scores_one_without_judging(self, tmp_path):
        case = EvalCase(question="q", generated_answer="   ", retrieved_chunks=())
        judge = make_judge(tmp_path, {}, default="should not be used")
        result = score_faithfulness(case, judge)
        assert result.score == 1.0
        assert result.raw_judge_output is None
        assert judge.provider.calls == 0  # judge never called

    def test_length_mismatch_raises(self, tmp_path):
        case = EvalCase(
            question="q",
            generated_answer="One. Two. Three.",  # 3 claims
            retrieved_chunks=(Chunk("c1", "ctx"),),
        )
        # Judge returns only 2 verdicts -> mismatch.
        verdict = '```json\n{"claims": [{"index": 0, "supported": true}, {"index": 1, "supported": true}]}\n```'
        judge = make_judge(tmp_path, {"One.": verdict})
        with pytest.raises(VerdictParseError, match="2 verdicts for 3 claims"):
            score_faithfulness(case, judge)

    def test_rationale_lists_each_claim(self, tmp_path):
        case = EvalCase(
            question="q",
            generated_answer="Good claim. Bad claim.",
            retrieved_chunks=(Chunk("c1", "ctx"),),
        )
        verdict = '```json\n{"claims": [{"index": 0, "supported": true}, {"index": 1, "supported": false}]}\n```'
        judge = make_judge(tmp_path, {"Good claim": verdict})
        result = score_faithfulness(case, judge)
        assert "✓ Good claim." in result.rationale
        assert "✗ Bad claim." in result.rationale


# --------------------------------------------------------------------------- #
# Stub integrity + end-to-end over the hand-written stubs                     #
# --------------------------------------------------------------------------- #
class TestStubs:
    def test_label_count_matches_claim_count(self):
        # Guards against a stub's labels drifting out of sync with its sentences.
        for stub in STUBS:
            n_claims = len(decompose_claims(stub.case.generated_answer))
            assert len(stub.label) == n_claims, stub.case.question

    def test_fake_keys_are_unique(self):
        # No collision => each stub gets its own canned verdict.
        assert len(FAKE_RESPONSES) == len(STUBS)

    @pytest.mark.parametrize("stub", STUBS, ids=lambda s: s.case.question[:30])
    def test_each_stub_scores_to_its_label_fraction(self, tmp_path, stub):
        judge = make_judge(
            tmp_path, FAKE_RESPONSES, default='```json\n{"claims": []}\n```'
        )
        result = score_faithfulness(stub.case, judge)
        expected = sum(stub.label) / len(stub.label)
        assert result.score == pytest.approx(expected)
