from __future__ import annotations

from evalrag.runner.adapter import FakePipelineAdapter
from evalrag.runner.loop import run_pipeline
from testdata.runner_inputs import INPUTS, RAISES, RESPONSES


def main() -> None:
    adapter = FakePipelineAdapter(responses=RESPONSES, raises=RAISES)
    result = run_pipeline(INPUTS, adapter)

    print(f"=== Runner: {len(result.cases)} cases, {len(result.errors)} errors ===\n")
    for i, case in enumerate(result.cases, 1):
        print(f"Case {i}: {case.question}")
        print(f"  answer: {case.generated_answer}")
        print(f"  chunks: {[c.id for c in case.retrieved_chunks]}")
        print(f"  expected_answer: {case.expected_answer}")
        print(f"  source_chunk_id: {case.source_chunk_id}\n")

    for err in result.errors:
        print(f"ERROR on {err.question!r}: {err.error}")


if __name__ == "__main__":
    main()
