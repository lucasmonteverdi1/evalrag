from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Return the model's raw text completion for `prompt`."""
        ...


class FakeProvider:
    """Deterministic test double. No network, no API key.

    `responses` maps a substring -> canned reply; the first substring found in
    the prompt wins. Falls back to `default`. Records call count so tests can
    assert the cache short-circuits real calls.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "",
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.calls = 0

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        self.calls += 1
        for needle, reply in self.responses.items():
            if needle in prompt:
                return reply
        return self.default