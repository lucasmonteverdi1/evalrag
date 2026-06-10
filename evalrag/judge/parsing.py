from __future__ import annotations

import json
import re
from typing import Any

# Finds fenced ```json ... ``` blocks (object or array). We take the LAST match
# so the judge's chain-of-thought can mention JSON earlier without breaking parsing.
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


class VerdictParseError(ValueError):
    """Raised when a judge's output has no parseable verdict block."""


def extract_last_json_block(raw: str) -> Any:
    """Parse the LAST fenced JSON block in the judge's raw output.

    Raises VerdictParseError if there is no block or it is not valid JSON.
    Callers are responsible for reading the fields they expect from the result.
    """
    matches = list(_JSON_BLOCK.finditer(raw))
    if not matches:
        raise VerdictParseError("no JSON verdict block found in judge output")
    blob = matches[-1].group(1)
    try:
        return json.loads(blob)
    except json.JSONDecodeError as e:
        raise VerdictParseError(f"malformed verdict block: {e}") from e
