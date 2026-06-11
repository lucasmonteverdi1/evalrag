from __future__ import annotations

from dataclasses import dataclass

from evalrag.judge.llm_client import LLMClient
from evalrag.scorer.faithfulness import judge_claims
from evalrag.validation.agreement import (
    Confusion,
    agreement_pct,
    cohens_kappa,
    confusion,
    f1,
)
from testdata.stubs import MetricStub


@dataclass(frozen=True)
class StubAgreement:
    """Per-stub breakdown, for traceability."""

    question: str
    human: list[bool]
    judge: list[bool]


@dataclass(frozen=True)
class AgreementReport:
    """Judge-vs-human agreement over a holdout, at the per-claim level."""

    n_claims: int
    agreement_pct: float
    f1: float
    kappa: float
    confusion: Confusion
    per_stub: list[StubAgreement]


def collect_pairs(
    stubs: list[MetricStub[list[bool]]],
    judge: LLMClient,
    *,
    prompts_dir: str = "prompts",
) -> tuple[list[bool], list[bool], list[StubAgreement]]:
    """Run the judge over each stub and align its per-claim verdicts with the
    human labels.

    Returns two flat, position-aligned bool lists (human, judge) spanning all
    claims of all stubs, plus a per-stub breakdown. Raises ValueError if a stub's
    human label count doesn't match the number of claims the judge scored (a sign
    the stub's labels drifted out of sync with its answer).
    """
    human_all: list[bool] = []
    judge_all: list[bool] = []
    per_stub: list[StubAgreement] = []

    for stub in stubs:
        _claims, verdicts, _raw = judge_claims(stub.case, judge, prompts_dir=prompts_dir)
        if len(stub.label) != len(verdicts):
            raise ValueError(
                f"stub {stub.case.question!r}: {len(stub.label)} human labels "
                f"but judge produced {len(verdicts)} verdicts"
            )
        human_all.extend(stub.label)
        judge_all.extend(verdicts)
        per_stub.append(
            StubAgreement(
                question=stub.case.question,
                human=list(stub.label),
                judge=list(verdicts),
            )
        )

    return human_all, judge_all, per_stub


def validate_faithfulness(
    stubs: list[MetricStub[list[bool]]],
    judge: LLMClient,
    *,
    prompts_dir: str = "prompts",
) -> AgreementReport:
    """Compute the per-claim agreement report for the faithfulness judge over
    the given hand-labeled stubs."""
    human, judge_labels, per_stub = collect_pairs(stubs, judge, prompts_dir=prompts_dir)
    return AgreementReport(
        n_claims=len(human),
        agreement_pct=agreement_pct(human, judge_labels),
        f1=f1(human, judge_labels),
        kappa=cohens_kappa(human, judge_labels),
        confusion=confusion(human, judge_labels),
        per_stub=per_stub,
    )
