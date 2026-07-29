
from evalrag.config import ProviderConfig
from evalrag.judge.cache import ResponseCache
from evalrag.judge.llm_client import LLMClient
from evalrag.judge.provider import FakeProvider
from evalrag.scorer.faithfulness import decompose_claims
from evalrag.validation.agreement import Confusion
from evalrag.validation.harness import (
    AgreementReport,
    collect_pairs,
    validate_faithfulness,
)
from testdata.faithfulness_stubs import FAKE_RESPONSES, STUBS


def make_judge(tmp_path, responses) -> LLMClient:
    return LLMClient(
        provider=FakeProvider(responses=responses, default='```json\n{"claims": []}\n```'),
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


def total_claims() -> int:
    return sum(len(decompose_claims(s.case.generated_answer)) for s in STUBS)


class TestValidateFaithfulness:
    def test_perfect_agreement_when_judge_echoes_labels(self, tmp_path):
        # FAKE_RESPONSES is generated from the human labels, so judge == human.
        judge = make_judge(tmp_path, FAKE_RESPONSES)
        report = validate_faithfulness(STUBS, judge)
        assert isinstance(report, AgreementReport)
        assert report.agreement_pct == 1.0
        assert report.f1 == 1.0
        assert report.kappa == 1.0

    def test_n_claims_matches_total(self, tmp_path):
        judge = make_judge(tmp_path, FAKE_RESPONSES)
        report = validate_faithfulness(STUBS, judge)
        assert report.n_claims == total_claims()

    def test_per_stub_breakdown_present(self, tmp_path):
        judge = make_judge(tmp_path, FAKE_RESPONSES)
        report = validate_faithfulness(STUBS, judge)
        assert len(report.per_stub) == len(STUBS)
        for stub, breakdown in zip(STUBS, report.per_stub):
            assert breakdown.question == stub.case.question
            assert breakdown.human == stub.label
            assert breakdown.judge == stub.label  # judge echoes labels here

    def test_confusion_is_all_true_negatives_and_positives(self, tmp_path):
        # With perfect agreement there are no fp/fn.
        judge = make_judge(tmp_path, FAKE_RESPONSES)
        report = validate_faithfulness(STUBS, judge)
        assert isinstance(report.confusion, Confusion)
        assert report.confusion.fp == 0
        assert report.confusion.fn == 0
        assert report.confusion.total == total_claims()


class TestCollectPairs:
    def test_returns_aligned_flat_lists(self, tmp_path):
        judge = make_judge(tmp_path, FAKE_RESPONSES)
        human, judge_labels, per_stub = collect_pairs(STUBS, judge)
        assert len(human) == len(judge_labels) == total_claims()
        assert len(per_stub) == len(STUBS)

    def test_skips_stub_on_count_drift(self, tmp_path):
        # A judge whose verdict count mismatches the labels: that stub is skipped
        # (not crashed), so the run continues with the remaining stubs.
        bad = make_judge(tmp_path, {})  # default -> 0 verdicts for everything
        multi = [s for s in STUBS if len(s.label) >= 1][:1]
        human, judge_labels, per_stub = collect_pairs(multi, bad)
        assert per_stub == []  # the drifting stub was dropped, no exception

    def test_unparseable_reply_skips_not_crashes(self, tmp_path):
        # The real-world case: judge answers in prose, no JSON. Skip that stub.
        prose = make_judge(tmp_path, {"__none__": "x"})  # default has no JSON block
        prose.provider.default = "The verdict is: true (no JSON here)."
        human, judge_labels, per_stub = collect_pairs(STUBS[:1], prose)
        assert per_stub == []
