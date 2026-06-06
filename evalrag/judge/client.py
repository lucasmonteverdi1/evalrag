from __future__ import annotations

from dataclasses import dataclass

from evalrag.config import ProviderConfig
from evalrag.judge.cache import ResponseCache
from evalrag.judge.provider import LLMProvider


@dataclass(frozen=True)
class JudgeResult:
    raw_output: str
    cached: bool


class JudgeClient:
    def __init__(
        self,
        provider: LLMProvider,
        config: ProviderConfig,
        cache: ResponseCache,
        prompt_versions: dict[str, str],
    ) -> None:
        self.provider = provider
        self.config = config
        self.cache = cache
        self.prompt_versions = prompt_versions

    def version_for(self, metric: str) -> str:
        if metric not in self.prompt_versions:
            raise KeyError(f"no pinned prompt version for metric '{metric}'")
        return self.prompt_versions[metric]

    def judge(self, metric: str, rendered_prompt: str) -> JudgeResult:
        version = self.version_for(metric)
        key = self.cache.make_key(
            provider=self.config.provider,
            model=self.config.model,
            prompt_version=f"{metric}:{version}",
            prompt_text=rendered_prompt,
        )
        hit = self.cache.get(key)
        if hit is not None:
            return JudgeResult(raw_output=hit, cached=True)

        raw = self.provider.complete(
            rendered_prompt,
            model=self.config.model,
            temperature=self.config.temperature,   # config pins this to 0
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_seconds,
        )
        self.cache.set(
            key,
            raw,
            meta={"metric": metric, "version": version, "model": self.config.model},
        )
        return JudgeResult(raw_output=raw, cached=False)