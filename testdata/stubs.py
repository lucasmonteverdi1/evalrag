from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from evalrag.types import EvalCase

# Allows the "label" variable to be generic.
L = TypeVar("L")


@dataclass(frozen=True)
class MetricStub(Generic[L]):
    """A handwritten eval case plus its hand-labeled ground truth.

    The `label` shape varies by metric:
      - faithfulness:      list[bool]   (one per claim)
      - context_precision: list[bool]   (one per retrieved chunk)
      - answer_relevance:  bool / float (one verdict for the whole answer)
    """

    case: EvalCase
    label: L
