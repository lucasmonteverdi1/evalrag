import pytest

from evalrag.validation.agreement import (
    Confusion,
    LengthMismatchError,
    agreement_pct,
    cohens_kappa,
    confusion,
    f1,
)


class TestConfusion:
    def test_counts_each_cell(self):
        human = [True, True, False, False]
        judge = [True, False, True, False]
        c = confusion(human, judge)
        assert c == Confusion(tp=1, fp=1, tn=1, fn=1)
        assert c.total == 4

    def test_length_mismatch_raises(self):
        with pytest.raises(LengthMismatchError):
            confusion([True], [True, False])


class TestAgreementPct:
    def test_perfect(self):
        assert agreement_pct([True, False, True], [True, False, True]) == 1.0

    def test_total_disagreement(self):
        assert agreement_pct([True, True], [False, False]) == 0.0

    def test_partial(self):
        # 3 of 4 match.
        assert agreement_pct([True, True, False, False], [True, True, True, False]) == 0.75

    def test_empty_is_one(self):
        assert agreement_pct([], []) == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(LengthMismatchError):
            agreement_pct([True], [])


class TestF1:
    def test_perfect(self):
        assert f1([True, False, True], [True, False, True]) == 1.0

    def test_known_value(self):
        # tp=2, fp=1, fn=1 -> F1 = 2*2 / (2*2 + 1 + 1) = 4/6
        human = [True, True, True, False]
        judge = [True, True, False, True]
        assert f1(human, judge) == pytest.approx(4 / 6)

    def test_no_positives_anywhere_is_one(self):
        # Documented degenerate case: tp=fp=fn=0 -> 1.0
        assert f1([False, False], [False, False]) == 1.0

    def test_judge_all_wrong_positives(self):
        # human all False, judge all True -> tp=0, fp=2, fn=0 -> denom=2 -> 0.0
        assert f1([False, False], [True, True]) == 0.0


class TestCohensKappa:
    def test_perfect_agreement(self):
        assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0

    def test_chance_level_is_zero(self):
        # A worked example where observed agreement equals chance agreement.
        # human: 2 True / 2 False; judge: 2 True / 2 False; po such that kappa=0.
        # human=[T,T,F,F], judge=[T,F,T,F]: po=0.5, marginals 0.5/0.5 -> pe=0.5 -> kappa=0
        assert cohens_kappa([True, True, False, False], [True, False, True, False]) == 0.0

    def test_negative_when_worse_than_chance(self):
        # human=[T,F,T,F], judge=[F,T,F,T]: po=0, pe=0.5 -> kappa=-1
        assert cohens_kappa([True, False, True, False], [False, True, False, True]) == -1.0

    def test_constant_labels_full_agreement(self):
        # Both rate everything True (pe == 1). Full agreement -> sentinel 1.0
        assert cohens_kappa([True, True, True], [True, True, True]) == 1.0

    def test_constant_labels_disagreement(self):
        # human all True, judge all False (pe == 1, po == 0) -> sentinel 0.0
        assert cohens_kappa([True, True], [False, False]) == 0.0

    def test_empty_is_one(self):
        assert cohens_kappa([], []) == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(LengthMismatchError):
            cohens_kappa([True], [True, False])
