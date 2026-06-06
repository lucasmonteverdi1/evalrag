from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam


class OpenRouterProvider:
    """LLMProvider backed by the OpenAI SDK pointed at any OpenAI-compatible
    endpoint (OpenRouter by default). The api_key is resolved by the caller
    (config.resolve_api_key), this class never touches the environment.
    """

    def __init__(self, api_key: str, base_url: str, max_retries: int = 3) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                ChatCompletionUserMessageParam(content=prompt, role="user"),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""