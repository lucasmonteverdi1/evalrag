import textwrap

import pytest

from evalrag.config import (
    CacheConfig,
    ConfigError,
    ModelsConfig,
    ProviderConfig,
    judge_matches_sut,
    load_models_config,
    load_prompt_versions,
    resolve_api_key,
)

VALID_MODELS_YAML = textwrap.dedent(
    """
    judge:
      provider: openrouter
      base_url: https://openrouter.ai/api/v1
      api_key_env: OPENROUTER_API_KEY
      model: google/gemini-2.5-flash
      temperature: 0
      max_tokens: 1024
      timeout_seconds: 30
      max_retries: 3
    system_under_test:
      provider: openrouter
      base_url: https://openrouter.ai/api/v1
      api_key_env: OPENROUTER_API_KEY
      model: openai/gpt-4o
      temperature: 0
      max_tokens: 512
    cache:
      enabled: true
      dir: .cache/llm_responses
    """
)


def _write(tmp_path, text: str):
    path = tmp_path / "models.yaml"
    path.write_text(text)
    return path


class TestLoadModelsConfig:
    def test_parses_valid_yaml(self, tmp_path):
        cfg = load_models_config(_write(tmp_path, VALID_MODELS_YAML))
        assert isinstance(cfg, ModelsConfig)
        assert cfg.judge.model == "google/gemini-2.5-flash"
        assert cfg.judge.temperature == 0.0
        assert cfg.judge.max_tokens == 1024
        assert cfg.system_under_test.model == "openai/gpt-4o"
        assert cfg.cache.enabled is True
        assert cfg.cache.dir == ".cache/llm_responses"

    def test_defaults_applied(self, tmp_path):
        # system_under_test omits timeout_seconds/max_retries -> defaults.
        cfg = load_models_config(_write(tmp_path, VALID_MODELS_YAML))
        assert cfg.system_under_test.timeout_seconds == 30
        assert cfg.system_under_test.max_retries == 3

    def test_cache_defaults_when_section_absent(self, tmp_path):
        text = VALID_MODELS_YAML.split("cache:")[0]  # drop the cache block
        cfg = load_models_config(_write(tmp_path, text))
        assert cfg.cache == CacheConfig()

    def test_real_committed_config_parses(self):
        # Guards against drift between configs/models.yaml and the dataclasses.
        cfg = load_models_config("configs/models.yaml")
        assert cfg.judge.base_url
        assert cfg.judge.api_key_env
        assert cfg.system_under_test.model


class TestValidation:
    def test_missing_section_raises(self, tmp_path):
        # judge present but system_under_test absent.
        text = textwrap.dedent(
            """
            judge:
              provider: openrouter
              base_url: https://x
              api_key_env: K
              model: m
              temperature: 0
              max_tokens: 10
            """
        )
        with pytest.raises(ConfigError, match="system_under_test"):
            load_models_config(_write(tmp_path, text))

    def test_missing_required_key_raises(self):
        with pytest.raises(ConfigError, match="missing required keys"):
            ProviderConfig.from_dict(
                {"provider": "openrouter", "base_url": "https://x"},  # missing model etc.
                section="judge",
            )

    def test_non_numeric_temperature_raises(self):
        with pytest.raises(ConfigError, match="temperature must be a number"):
            ProviderConfig.from_dict(
                {
                    "provider": "openrouter",
                    "base_url": "https://x",
                    "api_key_env": "K",
                    "model": "m",
                    "temperature": "hot",
                    "max_tokens": 10,
                },
                section="judge",
            )

    def _valid(self, **overrides):
        data = {
            "provider": "openrouter",
            "base_url": "https://x",
            "api_key_env": "K",
            "model": "m",
            "temperature": 0,
            "max_tokens": 10,
        }
        data.update(overrides)
        return data

    def test_bad_base_url_raises(self):
        with pytest.raises(ConfigError, match="base_url must start with"):
            ProviderConfig.from_dict(self._valid(base_url="ftp://x"), section="judge")

    def test_temperature_out_of_range_raises(self):
        with pytest.raises(ConfigError, match=r"temperature must be in \[0, 2\]"):
            ProviderConfig.from_dict(self._valid(temperature=3.0), section="judge")

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ConfigError, match="max_tokens must be >= 1"):
            ProviderConfig.from_dict(self._valid(max_tokens=0), section="judge")

    def test_negative_timeout_raises(self):
        with pytest.raises(ConfigError, match="timeout_seconds must be >= 1"):
            ProviderConfig.from_dict(self._valid(timeout_seconds=-5), section="judge")

    def test_zero_max_retries_allowed(self):
        # max_retries == 0 is valid (no retries); floor is 0, not 1.
        cfg = ProviderConfig.from_dict(self._valid(max_retries=0), section="judge")
        assert cfg.max_retries == 0


class TestOverridesPrecedence:
    def test_override_beats_yaml(self, tmp_path):
        cfg = load_models_config(
            _write(tmp_path, VALID_MODELS_YAML),
            overrides={"judge": {"model": "anthropic/claude-3.5-sonnet"}},
        )
        # Overridden key wins; sibling keys untouched.
        assert cfg.judge.model == "anthropic/claude-3.5-sonnet"
        assert cfg.judge.max_tokens == 1024

    def test_no_overrides_is_noop(self, tmp_path):
        cfg = load_models_config(_write(tmp_path, VALID_MODELS_YAML), overrides=None)
        assert cfg.judge.model == "google/gemini-2.5-flash"


class TestResolveApiKey:
    def test_returns_env_value(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
        cfg = ProviderConfig(
            provider="openrouter",
            base_url="https://x",
            api_key_env="OPENROUTER_API_KEY",
            model="m",
            temperature=0,
            max_tokens=10,
        )
        assert resolve_api_key(cfg) == "sk-test-123"

    def test_raises_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        cfg = ProviderConfig(
            provider="openrouter",
            base_url="https://x",
            api_key_env="OPENROUTER_API_KEY",
            model="m",
            temperature=0,
            max_tokens=10,
        )
        with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
            resolve_api_key(cfg)


class TestJudgeMatchesGenerator:
    def _cfg(self, judge_model, gen_model, judge_url="https://a", gen_url="https://a"):
        def mk(model, url):
            return ProviderConfig(
                provider="openrouter",
                base_url=url,
                api_key_env="K",
                model=model,
                temperature=0,
                max_tokens=10,
            )

        return ModelsConfig(
            judge=mk(judge_model, judge_url),
            system_under_test=mk(gen_model, gen_url),
            cache=CacheConfig(),
        )

    def test_true_when_same_model_and_url(self):
        assert judge_matches_sut(self._cfg("m", "m")) is True

    def test_false_when_different_model(self):
        assert judge_matches_sut(self._cfg("judge-m", "gen-m")) is False

    def test_false_when_same_model_different_url(self):
        cfg = self._cfg("m", "m", judge_url="https://a", gen_url="https://b")
        assert judge_matches_sut(cfg) is False


class TestLoadPromptVersions:
    def test_loads_real_committed_prompts(self):
        versions = load_prompt_versions("configs/prompts.yaml")
        assert versions["faithfulness"] == "v1"

    def test_coerces_values_to_str(self, tmp_path):
        path = tmp_path / "prompts.yaml"
        path.write_text("faithfulness: 1\n")  # YAML reads 1 as int
        versions = load_prompt_versions(path)
        assert versions["faithfulness"] == "1"
