from __future__ import annotations

from pathlib import Path

from evalrag.report.aggregate import CaseReport, aggregate
from evalrag.report.html_report import render_html
from evalrag.report.json_report import render_json
from evalrag.types import Chunk, EvalCase, MetricResult

# A couple of stub evaluated cases (what the orchestrator would produce).
CASE_REPORTS = [
    CaseReport(
        case=EvalCase(
            question="What is the capital of France?",
            generated_answer="Paris is the capital of France.",
            retrieved_chunks=(Chunk("c1", "Paris is the capital of France."),),
            source_chunk_id="c1",
        ),
        results=[
            MetricResult("faithfulness", 1.0, "1/1 claims grounded.", '{"claims": []}'),
            MetricResult("answer_relevance", 0.9, "directly answers", None),
            MetricResult("context_precision", 1.0, "deterministic: p@k=1.00, MRR=1.00", None),
        ],
    ),
    CaseReport(
        case=EvalCase(
            question="How big is Tokyo?",
            generated_answer="Tokyo is in Brazil.",
            retrieved_chunks=(Chunk("c2", "Tokyo is the capital of Japan."),),
        ),
        results=[
            MetricResult("faithfulness", 0.0, "0/1 claims grounded.", None),
            MetricResult("answer_relevance", 0.2, "off topic", None),
        ],
    ),
]


def main() -> None:
    agg = aggregate(CASE_REPORTS)
    out_dir = Path(".cache/report-demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    html_path = out_dir / "report.html"
    json_path.write_text(render_json(CASE_REPORTS, agg))
    html_path.write_text(render_html(CASE_REPORTS, agg))

    print("=== Aggregate ===")
    for metric, score in agg.per_metric.items():
        print(f"  {metric}: {score:.3f}")
    print(f"  overall (informational): {agg.overall:.3f}")
    print(f"\nWrote {json_path} and {html_path}")


if __name__ == "__main__":
    main()
