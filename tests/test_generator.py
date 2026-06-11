import json

import pytest

from evalrag.config import ProviderConfig
from evalrag.generator.synthesize import (
    _parse_pairs,
    generate_for_chunk,
    generate_inputs,
)
from evalrag.judge.cache import ResponseCache
from evalrag.judge.llm_client import LLMClient
from evalrag.judge.parsing import VerdictParseError
from evalrag.judge.provider import FakeProvider
from evalrag.types import Chunk


def block(pairs: list[dict]) -> str:
    return f'```json\n{json.dumps({"pairs": pairs})}\n```'


class TestParsePairs:
    def test_parses_pairs(self):
        raw = block([{"question": "q1", "expected_answer": "a1"}])
        assert _parse_pairs(raw) == [("q1", "a1")]

    def test_multiple_pairs(self):
        raw = block(
            [
                {"question": "q1", "expected_answer": "a1"},
                {"question": "q2", "expected_answer": "a2"},
            ]
        )
        assert _parse_pairs(raw) == [("q1", "a1"), ("q2", "a2")]

    def test_no_block_raises(self):
        with pytest.raises(VerdictParseError, match="no JSON verdict block"):
            _parse_pairs("no json here")

    def test_missing_field_raises(self):
        raw = block([{"question": "q1"}])  # no expected_answer
        with pytest.raises(VerdictParseError, match="malformed"):
            _parse_pairs(raw)


def make_generator(tmp_path, responses) -> LLMClient:
    return LLMClient(
        provider=FakeProvider(responses=responses, default='```json\n{"pairs": []}\n```'),
        config=ProviderConfig(
            provider="fake",
            base_url="n/a",
            api_key_env="UNUSED",
            model="fake-generator",
            temperature=0,
            max_tokens=512,
        ),
        cache=ResponseCache(enabled=False, cache_dir=tmp_path),
        prompt_versions={"generation": "v1"},
    )


class TestGenerateForChunk:
    def test_wires_source_chunk_id(self, tmp_path):
        chunk = Chunk("c1", "Paris is the capital of France.")
        resp = block([{"question": "What is the capital of France?", "expected_answer": "Paris."}])
        gen = make_generator(tmp_path, {"Paris is the capital": resp})
        inputs = generate_for_chunk(chunk, gen, n=1)
        assert len(inputs) == 1
        assert inputs[0].question == "What is the capital of France?"
        assert inputs[0].expected_answer == "Paris."
        assert inputs[0].source_chunk_id == "c1"  # ground truth wired through


class TestGenerateInputs:
    def test_spans_all_chunks(self, tmp_path):
        chunks = [Chunk("c1", "Alpha fact."), Chunk("c2", "Beta fact.")]
        responses = {
            "Alpha fact.": block([{"question": "qa", "expected_answer": "aa"}]),
            "Beta fact.": block([{"question": "qb", "expected_answer": "ab"}]),
        }
        gen = make_generator(tmp_path, responses)
        inputs = generate_inputs(chunks, gen, n_per_chunk=1)
        assert len(inputs) == 2
        assert {i.source_chunk_id for i in inputs} == {"c1", "c2"}

    def test_count_scales_with_n_per_chunk(self, tmp_path):
        chunk = Chunk("c1", "Multi fact.")
        resp = block(
            [
                {"question": "q1", "expected_answer": "a1"},
                {"question": "q2", "expected_answer": "a2"},
            ]
        )
        gen = make_generator(tmp_path, {"Multi fact.": resp})
        inputs = generate_inputs([chunk], gen, n_per_chunk=2)
        assert len(inputs) == 2
