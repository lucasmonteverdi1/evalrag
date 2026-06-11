import json

from evalrag.report.aggregate import CaseReport, aggregate
from evalrag.report.html_report import render_html
from evalrag.report.json_report import build_report, render_json
from evalrag.types import Chunk, EvalCase, MetricResult


def case(question="q", answer="a", chunks=(("c1", "x"),), **kw):
    return EvalCase(
        question=question,
        generated_answer=answer,
        retrieved_chunks=tuple(Chunk(i, t) for i, t in chunks),
        **kw,
    )


def cr(results, **kw):
    return CaseReport(case=case(**kw), results=results)


class TestAggregate:
    def test_mean_per_metric(self):
        reports = [
            cr([MetricResult("faithfulness", 1.0, "r")]),
            cr([MetricResult("faithfulness", 0.0, "r")]),
        ]
        agg = aggregate(reports)
        assert agg.per_metric["faithfulness"] == 0.5
        assert agg.n_cases == 2

    def test_macro_average_overall(self):
        # faithfulness mean = 1.0 (one case), relevance mean = 0.0 (one case).
        # Macro overall = (1.0 + 0.0) / 2 = 0.5, regardless of case counts.
        reports = [
            cr([MetricResult("faithfulness", 1.0, "r")]),
            cr([MetricResult("faithfulness", 1.0, "r"), MetricResult("answer_relevance", 0.0, "r")]),
        ]
        agg = aggregate(reports)
        assert agg.per_metric["faithfulness"] == 1.0
        assert agg.per_metric["answer_relevance"] == 0.0
        assert agg.overall == 0.5  # macro, not (1+1+0)/3

    def test_empty_overall_is_none(self):
        agg = aggregate([])
        assert agg.per_metric == {}
        assert agg.overall is None
        assert agg.n_cases == 0

    def test_case_with_no_results_counts_but_adds_no_scores(self):
        agg = aggregate([cr([])])
        assert agg.n_cases == 1
        assert agg.per_metric == {}


class TestJsonReport:
    def test_round_trips_and_has_traceability(self):
        reports = [
            cr(
                [MetricResult("faithfulness", 0.5, "half grounded", '{"raw": 1}')],
                question="What is X?",
                source_chunk_id="c1",
            )
        ]
        agg = aggregate(reports)
        text = render_json(reports, agg)
        data = json.loads(text)  # valid JSON

        assert data["summary"]["per_metric"]["faithfulness"] == 0.5
        assert data["summary"]["overall"] == 0.5
        case0 = data["cases"][0]
        assert case0["question"] == "What is X?"
        assert case0["source_chunk_id"] == "c1"
        assert case0["metrics"][0]["raw_judge_output"] == '{"raw": 1}'

    def test_build_report_overall_none_serializes(self):
        report = build_report([], aggregate([]))
        assert report["summary"]["overall"] is None
        assert json.loads(json.dumps(report))["summary"]["overall"] is None


class TestHtmlReport:
    def test_contains_scores_and_rationale(self):
        reports = [cr([MetricResult("faithfulness", 0.75, "3/4 grounded")])]
        agg = aggregate(reports)
        out = render_html(reports, agg)
        assert "faithfulness" in out
        assert "0.750" in out
        assert "3/4 grounded" in out
        assert "informational" in out  # overall is labeled non-gating

    def test_escapes_html_in_content(self):
        # A rationale with HTML must be escaped, not injected.
        reports = [cr([MetricResult("faithfulness", 1.0, "<script>alert(1)</script>")])]
        out = render_html(reports, aggregate(reports))
        assert "<script>alert(1)</script>" not in out
        assert "&lt;script&gt;" in out

    def test_overall_none_renders_na(self):
        out = render_html([], aggregate([]))
        assert "n/a" in out

    def test_raw_output_in_details_when_present(self):
        reports = [cr([MetricResult("faithfulness", 1.0, "r", "RAW_JUDGE_TEXT")])]
        out = render_html(reports, aggregate(reports))
        assert "RAW_JUDGE_TEXT" in out
        assert "<details>" in out
