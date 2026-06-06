from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import yaml

class ConfigError(ValueError):
    """Raised when a config file is missing required keys or has bad values."""


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: str
    api_key_env: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int = 30
    max_retries: int = 3

    @classmethod
    def from_dict(cls, data: dict, *, section: str) -> "ProviderConfig":
        required = ("provider", "base_url", "api_key_env", "model", "temperature", "max_tokens")
        missing = [k for k in required if k not in data]
        if missing:
            raise ConfigError(f"{section}: missing required keys: {', '.join(missing)}")
        if not isinstance(data["temperature"], (int, float)):
            raise ConfigError(f"{section}: temperature must be a number")
        return cls(
            provider=data["provider"],
            base_url=data["base_url"],
            api_key_env=data["api_key_env"],
            model=data["model"],
            temperature=float(data["temperature"]),
            max_tokens=int(data["max_tokens"]),
            timeout_seconds=int(data.get("timeout_seconds", 30)),
            max_retries=int(data.get("max_retries", 3)),
        )


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    dir: str = ".cache/llm_responses"

    @classmethod
    def from_dict(cls, data: dict) -> "CacheConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            dir=str(data.get("dir", ".cache/llm_responses")),
        )


@dataclass(frozen=True)
class ModelsConfig:
    judge: ProviderConfig
    pipeline_generator: ProviderConfig
    cache: CacheConfig

    @classmethod
    def from_dict(cls, data: dict) -> "ModelsConfig":
        for key in ("judge", "pipeline_generator"):
            if key not in data:
                raise ConfigError(f"models config: missing required section '{key}'")
        return cls(
            judge=ProviderConfig.from_dict(data["judge"], section="judge"),
            pipeline_generator=ProviderConfig.from_dict(
                data["pipeline_generator"], section="pipeline_generator"
            ),
            cache=CacheConfig.from_dict(data.get("cache", {})),
        )

def _deep_merge(base: dict, overrides: dict | None) -> dict:
    """Overlay `overrides` onto `base` (CLI flags beating YAML). Returns a new dict."""
    if not overrides:
        return base
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_models_config(
    path: str | Path = "configs/models.yaml",
    *,
    overrides: dict | None = None,
) -> ModelsConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    merged = _deep_merge(raw, overrides)   # precedence: CLI(overrides) > YAML
    return ModelsConfig.from_dict(merged)


def resolve_api_key(cfg: ProviderConfig) -> str:
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise ConfigError(
            f"API key not found: set the {cfg.api_key_env} environment variable"
        )
    return key


def judge_matches_generator(cfg: ModelsConfig) -> bool:
    """True when judge and pipeline generator are the same model on the same endpoint.

    The CLI uses this to warn about self-preference bias.
    """
    return (
        cfg.judge.model == cfg.pipeline_generator.model
        and cfg.judge.base_url == cfg.pipeline_generator.base_url
    )


def load_prompt_versions(path: str | Path = "configs/prompts.yaml") -> dict[str, str]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return {str(k): str(v) for k, v in raw.items()}