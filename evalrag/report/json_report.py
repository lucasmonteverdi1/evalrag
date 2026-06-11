from __future__ import annotations

import json

from evalrag.report.aggregate import Aggregate, CaseReport


def build_report(case_reports: list[CaseReport], agg: Aggregate) -> dict:
    """Build a plain-dict report (JSON-serializable) from cases + their aggregate.

    Captures everything needed for traceability: per-metric means, the overall
    summary, and every case's metric scores with rationale and raw judge output.
    """
    return {
        "summary": {
            "n_cases": agg.n_cases,
            "per_metric": agg.per_metric,
            "overall": agg.overall,  # informational only
        },
        "cases": [
            {
                "question": cr.case.question,
                "generated_answer": cr.case.generated_answer,
                "retrieved_chunk_ids": [c.id for c in cr.case.retrieved_chunks],
                "expected_answer": cr.case.expected_answer,
                "source_chunk_id": cr.case.source_chunk_id,
                "metrics": [
                    {
                        "metric": r.metric,
                        "score": r.score,
                        "rationale": r.rationale,
                        "raw_judge_output": r.raw_judge_output,
                    }
                    for r in cr.results
                ],
            }
            for cr in case_reports
        ],
    }


def render_json(case_reports: list[CaseReport], agg: Aggregate, *, indent: int = 2) -> str:
    """Serialize the report to a JSON string."""
    return json.dumps(build_report(case_reports, agg), indent=indent, ensure_ascii=False)
