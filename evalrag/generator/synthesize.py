from __future__ import annotations

from evalrag.judge.llm_client import LLMClient
from evalrag.judge.parsing import VerdictParseError, extract_last_json_block
from evalrag.judge.prompts import load_prompt
from evalrag.runner.loop import EvalInput
from evalrag.types import Chunk

METRIC = "generation"  # prompt key: prompts/generation_v1.md


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    """Extract (question, expected_answer) pairs from the LLM's raw output.

    Expects the LAST fenced JSON block: {"pairs": [{"question": "...",
    "expected_answer": "..."}, ...]}. Raises VerdictParseError if malformed.
    """
    data = extract_last_json_block(raw)
    try:
        pairs = data["pairs"]
        return [(str(p["question"]), str(p["expected_answer"])) for p in pairs]
    except (KeyError, TypeError) as e:
        raise VerdictParseError(f"malformed generation output: {e}") from e


def generate_for_chunk(
    chunk: Chunk,
    generator: LLMClient,
    *,
    n: int,
    prompts_dir: str | None = None,
) -> list[EvalInput]:
    """Generate n synthetic EvalInputs grounded in a single chunk.

    Each EvalInput records source_chunk_id=chunk.id so reference-based metrics
    (e.g. context precision's deterministic path) have ground truth.
    """
    version = generator.version_for(METRIC)
    template = load_prompt(METRIC, version, prompts_dir=prompts_dir)
    prompt = template.format(n=n, chunk=chunk.text)

    result = generator.judge(METRIC, prompt)
    pairs = _parse_pairs(result.raw_output)

    return [
        EvalInput(
            question=q,
            expected_answer=a,
            source_chunk_id=chunk.id,
        )
        for q, a in pairs
    ]


def generate_inputs(
    chunks: list[Chunk],
    generator: LLMClient,
    *,
    n_per_chunk: int = 1,
    prompts_dir: str | None = None,
) -> list[EvalInput]:
    """Generate synthetic EvalInputs across a set of source chunks."""
    inputs: list[EvalInput] = []
    for chunk in chunks:
        inputs.extend(
            generate_for_chunk(chunk, generator, n=n_per_chunk, prompts_dir=prompts_dir)
        )
    return inputs