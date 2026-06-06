import json

from evalrag.judge.cache import ResponseCache

KEY_ARGS = dict(
    provider="openrouter",
    model="google/gemini-2.5-flash",
    prompt_version="faithfulness:v1",
    prompt_text="some rendered prompt",
)


class TestMakeKey:
    def test_stable_for_same_inputs(self):
        k1 = ResponseCache.make_key(**KEY_ARGS)
        k2 = ResponseCache.make_key(**KEY_ARGS)
        assert k1 == k2

    def test_changes_when_any_field_changes(self):
        base = ResponseCache.make_key(**KEY_ARGS)
        for field, value in [
            ("provider", "openai"),
            ("model", "openai/gpt-4o"),
            ("prompt_version", "faithfulness:v2"),
            ("prompt_text", "different prompt"),
        ]:
            assert ResponseCache.make_key(**{**KEY_ARGS, field: value}) != base

    def test_field_boundaries_are_unambiguous(self):
        # "ab" + "c" must not collide with "a" + "bc" thanks to the separator.
        a = ResponseCache.make_key("ab", "c", "v", "p")
        b = ResponseCache.make_key("a", "bc", "v", "p")
        assert a != b


class TestGetSet:
    def test_miss_returns_none(self, tmp_path):
        cache = ResponseCache(enabled=True, cache_dir=tmp_path)
        assert cache.get("nonexistent") is None

    def test_set_then_get_roundtrip(self, tmp_path):
        cache = ResponseCache(enabled=True, cache_dir=tmp_path)
        key = ResponseCache.make_key(**KEY_ARGS)
        cache.set(key, "RAW RESPONSE", meta={"metric": "faithfulness"})
        assert cache.get(key) == "RAW RESPONSE"

    def test_stores_response_verbatim_with_meta(self, tmp_path):
        cache = ResponseCache(enabled=True, cache_dir=tmp_path)
        key = ResponseCache.make_key(**KEY_ARGS)
        cache.set(key, "verbatim\nmultiline", meta={"model": "x"})
        record = json.loads((tmp_path / f"{key}.json").read_text())
        assert record["response"] == "verbatim\nmultiline"
        assert record["meta"] == {"model": "x"}

    def test_disabled_get_always_none(self, tmp_path):
        cache = ResponseCache(enabled=False, cache_dir=tmp_path)
        key = ResponseCache.make_key(**KEY_ARGS)
        cache.set(key, "ignored")
        assert cache.get(key) is None

    def test_disabled_set_writes_nothing(self, tmp_path):
        cache = ResponseCache(enabled=False, cache_dir=tmp_path)
        cache.set("k", "v")
        assert list(tmp_path.iterdir()) == []
