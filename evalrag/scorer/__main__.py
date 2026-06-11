from __future__ import annotations

from evalrag.judge.cache import ResponseCache
from evalrag.judge.llm_client import LLMClient
from evalrag.judge.provider import FakeProvider
from evalrag.config import ProviderConfig
from evalrag.scorer.faithfulness import score_faithfulness
from testdata.faithfulness_stubs import STUBS, FAKE_RESPONSES


def main() -> None:
    provider = FakeProvider(responses=FAKE_RESPONSES, default='```json\n{"claims": []}\n```')
    config = ProviderConfig(
        provider="fake",
        base_url="n/a",
        api_key_env="UNUSED",
        model="fake-model",
        temperature=0,
        max_tokens=1024,
    )
    judge = LLMClient(
        provider=provider,
        config=config,
        cache=ResponseCache(enabled=False, cache_dir=".cache/demo"),
        prompt_versions={"faithfulness": "v1"},
    )

    for i, stub in enumerate(STUBS, 1):
        case = stub.case
        result = score_faithfulness(case, judge)
        print(f"\n=== Case {i}: {case.question}")
        print(f"score: {result.score:.2f}")
        print(result.rationale)


if __name__ == "__main__":
    main()