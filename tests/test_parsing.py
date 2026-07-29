import pytest

from evalrag.judge.parsing import VerdictParseError, extract_last_json_block


def test_fenced_block():
    raw = '```json\n{"score": 0.5}\n```'
    assert extract_last_json_block(raw) == {"score": 0.5}


def test_nested_object_in_fence():
    # The old non-greedy regex truncated at the first inner '}'.
    raw = '```json\n{"claims": [{"index": 0, "supported": true}]}\n```'
    assert extract_last_json_block(raw) == {"claims": [{"index": 0, "supported": True}]}


def test_bare_json_no_fences():
    # Real LLMs often drop the code fences.
    raw = 'The verdict is {"claims": [{"index": 0, "supported": true}]}'
    assert extract_last_json_block(raw) == {"claims": [{"index": 0, "supported": True}]}


def test_cot_then_bare_json_at_end():
    raw = "reasoning about index 0...\nFinal answer:\n{\"score\": 0.9}"
    assert extract_last_json_block(raw) == {"score": 0.9}


def test_bare_array():
    assert extract_last_json_block("result: [true, false]") == [True, False]


def test_last_wins():
    raw = '{"score": 0.1}\nactually:\n{"score": 0.9}'
    assert extract_last_json_block(raw) == {"score": 0.9}


def test_prose_inside_fence():
    # Model puts explanatory text before the JSON, inside the fence.
    raw = '```json\nHere is the verdict:\n{"score": 0.8}\n```'
    assert extract_last_json_block(raw) == {"score": 0.8}


def test_empty_fence_then_bare_json():
    raw = "```\n(nothing)\n```\nActual result: {\"score\": 0.7}"
    assert extract_last_json_block(raw) == {"score": 0.7}


def test_last_fence_wins_over_earlier():
    raw = '```json\n{"score": 0.1}\n```\nwait\n```json\n{"score": 0.9}\n```'
    assert extract_last_json_block(raw) == {"score": 0.9}


def test_ignores_chunk_refs_in_prose():
    # The real bug: CoT mentions [c1], [c2]; the bare JSON verdict must still win,
    # not a stray '[c1]' that isn't valid JSON.
    raw = (
        "0. Claim 0 [c1] supported.\n1. Claim 1 [c2] supported.\n"
        '{"claims": [{"index": 0, "supported": true}, {"index": 1, "supported": true}]}'
    )
    assert extract_last_json_block(raw) == {
        "claims": [{"index": 0, "supported": True}, {"index": 1, "supported": True}]
    }


def test_error_carries_raw():
    try:
        extract_last_json_block("no json [c1] here")
    except VerdictParseError as e:
        assert e.raw == "no json [c1] here"


def test_no_json_raises():
    with pytest.raises(VerdictParseError, match="no JSON"):
        extract_last_json_block("the model rambled without json")


def test_malformed_raises():
    with pytest.raises(VerdictParseError, match="malformed"):
        extract_last_json_block('```json\n{"score": tru }\n```')
