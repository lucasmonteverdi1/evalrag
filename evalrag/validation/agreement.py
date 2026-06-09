from __future__ import annotations

from dataclasses import dataclass


class LengthMismatchError(ValueError):
    """Raised when the human and judge label lists differ in length."""


@dataclass(frozen=True)
class Confusion:
    tp: int  # human=True,  judge=True
    fp: int  # human=False, judge=True
    tn: int  # human=False, judge=False
    fn: int  # human=True,  judge=False

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


def _check(human: list[bool], judge: list[bool]) -> None:
    if len(human) != len(judge):
        raise LengthMismatchError(
            f"length mismatch: {len(human)} human vs {len(judge)} judge labels"
        )


def confusion(human: list[bool], judge: list[bool]) -> Confusion:
    """Build the 2x2 confusion matrix. Positive class = True (claim supported)."""
    _check(human, judge)
    tp = fp = tn = fn = 0
    for h, j in zip(human, judge):
        if h and j:
            tp += 1
        elif not h and j:
            fp += 1
        elif not h and not j:
            tn += 1
        else:  # h and not j
            fn += 1
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def agreement_pct(human: list[bool], judge: list[bool]) -> float:
    """Fraction of claims where judge and human agree. Empty input -> 1.0."""
    _check(human, judge)
    if not human:
        return 1.0
    matches = sum(1 for h, j in zip(human, judge) if h == j)
    return matches / len(human)


def f1(human: list[bool], judge: list[bool]) -> float:
    """F1 with True (supported) as the positive class.

    Degenerate case: if there are no positives in either human or judge labels
    (tp == fp == fn == 0), F1 is defined here as 1.0 (perfect on a class with no
    instances) — documented choice, not a silent NaN.
    """
    c = confusion(human, judge)
    if c.tp == 0 and c.fp == 0 and c.fn == 0:
        return 1.0
    denom = 2 * c.tp + c.fp + c.fn
    if denom == 0:
        return 0.0
    return (2 * c.tp) / denom


def cohens_kappa(human: list[bool], judge: list[bool]) -> float:
    """Cohen's kappa: agreement corrected for chance.

    kappa = (po - pe) / (1 - pe), where po is observed agreement and pe is the
    agreement expected by chance. When pe == 1 (both raters gave a single constant
    label to everything) kappa is undefined; we return 1.0 if they fully agree,
    else 0.0 — a documented sentinel rather than a division by zero.
    """
    c = confusion(human, judge)
    n = c.total
    if n == 0:
        return 1.0
    po = (c.tp + c.tn) / n
    # Marginal probabilities for the chance-agreement term.
    p_yes = ((c.tp + c.fn) / n) * ((c.tp + c.fp) / n)
    p_no = ((c.tn + c.fp) / n) * ((c.tn + c.fn) / n)
    pe = p_yes + p_no
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)