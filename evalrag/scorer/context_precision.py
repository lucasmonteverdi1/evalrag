from __future__ import annotations

from evalrag.judge.llm_client import LLMClient
from evalrag.judge.parsing import VerdictParseError, extract_last_json_block
from evalrag.judge.prompts import load_prompt
from evalrag.types import EvalCase, MetricResult

METRIC = "context_precision"


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Fraction of retrieved chunks that are relevant. """
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids if cid in relevant_ids)
    return hits / len(retrieved_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean reciprocal rank: 1/rank of the FIRST relevant chunk (rank starts at 1).

    Rewards putting a relevant chunk high in the ranking. 0.0 if none are relevant.
    """
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _relevance_from_judge(
    case: EvalCase, judge: LLMClient, prompts_dir: str
) -> tuple[list[bool], str]:
    """Ask the judge to label each retrieved chunk relevant. Returns (verdicts, raw)."""
    version = judge.version_for(METRIC)
    template = load_prompt(METRIC, version, prompts_dir=prompts_dir)
    chunks_block = "\n\n".join(
        f"[{i}] {chunk.text}" for i, chunk in enumerate(case.retrieved_chunks)
    )
    prompt = template.format(question=case.question, chunks=chunks_block)

    result = judge.judge(METRIC, prompt)
    data = extract_last_json_block(result.raw_output)
    try:
        chunks = sorted(data["chunks"], key=lambda c: c["index"])
        verdicts = [bool(c["relevant"]) for c in chunks]
    except (KeyError, TypeError) as e:
        raise VerdictParseError(f"malformed verdict block: {e}") from e

    if len(verdicts) != len(case.retrieved_chunks):
        raise VerdictParseError(
            f"judge returned {len(verdicts)} verdicts for "
            f"{len(case.retrieved_chunks)} chunks"
        )
    return verdicts, result.raw_output


def score_context_precision(
    case: EvalCase,
    judge: LLMClient | None = None,
    *,
    prompts_dir: str = "prompts",
) -> MetricResult:
    """Score how well the retriever surfaced relevant chunks.

    NOTE: despite the name, the score is a *hybrid* of two ranking signals, not raw
    precision: it averages precision@k (how much of the retrieval was relevant) with
    MRR (how high the first relevant chunk ranked). This rewards both low noise and
    good ranking. Relevance comes from source_chunk_id when known (deterministic, no
    LLM), else from a per-chunk judge.
    """
    retrieved_ids = [chunk.id for chunk in case.retrieved_chunks]

    if case.source_chunk_id is not None:
        # Deterministic path: compare against the known relevant chunk ID. No LLM.
        relevant_ids = {case.source_chunk_id}
        mode = "deterministic (source_chunk_id)"
        raw_output = None
    else:
        # Judge path: ask the LLM which chunks are relevant, derive relevant IDs.
        if judge is None:
            raise ValueError(
                "context_precision needs a judge when source_chunk_id is absent"
            )
        verdicts, raw_output = _relevance_from_judge(case, judge, prompts_dir)
        relevant_ids = {cid for cid, ok in zip(retrieved_ids, verdicts) if ok}
        mode = "judge per-chunk"

    p_at_k = precision_at_k(retrieved_ids, relevant_ids)
    rank = mrr(retrieved_ids, relevant_ids)
    # Combine: precision rewards relevant-vs-noise; MRR rewards ranking relevant high.
    score = (p_at_k + rank) / 2

    rationale = (
        f"{mode}: precision@k={p_at_k:.2f}, MRR={rank:.2f} "
        f"over {len(retrieved_ids)} retrieved chunks."
    )
    return MetricResult(
        metric=METRIC,
        score=score,
        rationale=rationale,
        raw_judge_output=raw_output,
    )