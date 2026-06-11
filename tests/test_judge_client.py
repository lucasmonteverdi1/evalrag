import pytest

from evalrag.config import ProviderConfig
from evalrag.judge.cache import ResponseCache
from evalrag.judge.llm_client import LLMClient, JudgeResult
from evalrag.judge.provider import FakeProvider


def make_config(**overrides) -> ProviderConfig:
    base = dict(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model="google/gemini-2.5-flash",
        temperature=0,
        max_tokens=1024,
        timeout_seconds=30,
        max_retries=3,
    )
    base.update(overrides)
    return ProviderConfig(**base)


def make_client(tmp_path, provider, *, enabled=True, versions=None) -> LLMClient:
    return LLMClient(
        provider=provider,
        config=make_config(),
        cache=ResponseCache(enabled=enabled, cache_dir=tmp_path),
        prompt_versions=versions or {"faithfulness": "v1"},
    )


class TestVersionFor:
    def test_returns_pinned_version(self, tmp_path):
        client = make_client(tmp_path, FakeProvider())
        assert client.version_for("faithfulness") == "v1"

    def test_unknown_metric_raises(self, tmp_path):
        client = make_client(tmp_path, FakeProvider())
        with pytest.raises(KeyError, match="answer_relevance"):
            client.version_for("answer_relevance")


class TestJudge:
    def test_returns_raw_output_uncached_on_first_call(self, tmp_path):
        provider = FakeProvider(default="VERDICT JSON")
        client = make_client(tmp_path, provider)
        result = client.judge("faithfulness", "rendered prompt")
        assert isinstance(result, JudgeResult)
        assert result.raw_output == "VERDICT JSON"
        assert result.cached is False
        assert provider.calls == 1

    def test_second_call_hits_cache_without_provider(self, tmp_path):
        provider = FakeProvider(default="VERDICT JSON")
        client = make_client(tmp_path, provider)
        client.judge("faithfulness", "rendered prompt")
        result = client.judge("faithfulness", "rendered prompt")
        assert result.cached is True
        assert result.raw_output == "VERDICT JSON"
        assert provider.calls == 1  # provider NOT called again

    def test_different_prompt_is_a_separate_cache_entry(self, tmp_path):
        provider = FakeProvider(default="X")
        client = make_client(tmp_path, provider)
        client.judge("faithfulness", "prompt A")
        client.judge("faithfulness", "prompt B")
        assert provider.calls == 2

    def test_passes_config_params_to_provider(self, tmp_path):
        captured = {}

        class RecordingProvider:
            calls = 0

            def complete(self, prompt, *, model, temperature, max_tokens, timeout):
                captured.update(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                return "ok"

        client = make_client(tmp_path, RecordingProvider())
        client.judge("faithfulness", "the prompt")
        assert captured["model"] == "google/gemini-2.5-flash"
        assert captured["temperature"] == 0  # temp-0 guarantee
        assert captured["max_tokens"] == 1024
        assert captured["timeout"] == 30
        assert captured["prompt"] == "the prompt"

    def test_disabled_cache_recalls_provider(self, tmp_path):
        provider = FakeProvider(default="X")
        client = make_client(tmp_path, provider, enabled=False)
        client.judge("faithfulness", "p")
        result = client.judge("faithfulness", "p")
        assert provider.calls == 2
        assert result.cached is False

    def test_unknown_metric_raises_before_calling_provider(self, tmp_path):
        provider = FakeProvider(default="X")
        client = make_client(tmp_path, provider)
        with pytest.raises(KeyError):
            client.judge("answer_relevance", "p")
        assert provider.calls == 0
