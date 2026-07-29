from __future__ import annotations

import json
import re
from typing import Any

# Fenced ```json ... ``` block (non-greedy body).
_FENCED = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_DECODER = json.JSONDecoder()


class VerdictParseError(ValueError):
    """Raised when a judge's output has no parseable verdict block.

    Carries the raw judge output (`.raw`) so callers can log what actually came back.
    """

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


def _iter_json(text: str):
    """Yield every TOP-LEVEL JSON value in text, using stdlib's real decoder.

    raw_decode tells us where each value ends, so a stray '[c1]' in the judge's prose
    fails to decode and is skipped. We skip past a decoded value's span so nested
    objects (e.g. each {"index": ...} inside {"claims": [...]}) aren't yielded on
    their own — only the outer value counts.
    """
    pos = 0
    for m in re.finditer(r"[{\[]", text):
        if m.start() < pos:
            continue  # inside a value we already decoded
        try:
            value, end = _DECODER.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        pos = end
        yield value


def extract_last_json_block(raw: str) -> Any:
    """Parse a JSON verdict from the judge's raw output, tolerant of format drift.

    Prefers the last fenced ```json block; otherwise takes the LAST valid JSON value
    anywhere in the text (real LLMs drop fences and pad with chain-of-thought).
    """
    saw_fence = False
    for fenced in reversed(_FENCED.findall(raw)):
        saw_fence = True
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass  # fenced body had prose around the JSON; fall through to scanning

    values = list(_iter_json(raw))
    if values:
        return values[-1]  # last JSON value wins (trailing verdict over CoT)
    if saw_fence:
        raise VerdictParseError("malformed verdict block: no valid JSON found", raw=raw)
    raise VerdictParseError("no JSON verdict block found in judge output", raw=raw)
