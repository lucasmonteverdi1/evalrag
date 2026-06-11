from __future__ import annotations

import json

from evalrag.config import ProviderConfig
from evalrag.generator.synthesize import generate_inputs
from evalrag.judge.cache import ResponseCache
from evalrag.judge.llm_client import LLMClient
from evalrag.judge.provider import FakeProvider
from evalrag.types import Chunk

# A couple of source documents to generate questions from.
CHUNKS = [
    Chunk("c1", "Paris is the capital of France. The Seine river runs through it."),
    Chunk("c2", "The official language of Brazil is Portuguese."),
]

def _block(pairs: list[dict]) -> str:
    return f'```json\n{json.dumps({"pairs": pairs})}\n```'


# Canned generator output per chunk (keyed on a unique substring of each chunk's text).
FAKE_RESPONSES = {
    "Paris is the capital": _block(
        [{"question": "What is the capital of France?", "expected_answer": "Paris."}]
    ),
    "official language of Brazil": _block(
        [{"question": "What language is spoken in Brazil?", "expected_answer": "Portuguese."}]
    ),
}


def main() -> None:
    provider = FakeProvider(responses=FAKE_RESPONSES, default='```json\n{"pairs": []}\n```')
    # The generator uses the PIPELINE GENERATOR model config, not the judge's.
    config = ProviderConfig(
        provider="fake",
        base_url="n/a",
        api_key_env="UNUSED",
        model="fake-generator",
        temperature=0,
        max_tokens=512,
    )
    generator = LLMClient(
        provider=provider,
        config=config,
        cache=ResponseCache(enabled=False, cache_dir=".cache/gen-demo"),
        prompt_versions={"generation": "v1"},
    )

    inputs = generate_inputs(CHUNKS, generator, n_per_chunk=1)
    print(f"=== Generated {len(inputs)} EvalInputs ===\n")
    for inp in inputs:
        print(f"Q: {inp.question}")
        print(f"   expected: {inp.expected_answer}")
        print(f"   source_chunk_id: {inp.source_chunk_id}\n")


if __name__ == "__main__":
    main()
