from __future__ import annotations

import html
from string import Template

from evalrag.report.aggregate import Aggregate, CaseReport

_PAGE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EvalRAG Report</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
  h1 { margin-bottom: 0.25rem; }
  table { border-collapse: collapse; margin: 1rem 0; }
  th, td { border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; }
  th { background: #f3f3f3; }
  .overall { color: #666; font-style: italic; }
  .case { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin: 1rem 0; }
  .q { font-weight: 600; }
  .rationale { white-space: pre-wrap; color: #333; }
  details { margin-top: 0.4rem; }
  pre { background: #f7f7f7; padding: 0.6rem; overflow-x: auto; white-space: pre-wrap; }
</style>
</head>
<body>
<h1>EvalRAG Report</h1>
<p>$n_cases case(s) evaluated.</p>

<h2>Per-metric scores</h2>
<table>
<tr><th>Metric</th><th>Mean score</th></tr>
$metric_rows
</table>
<p class="overall">Overall (informational, not used for gating): $overall</p>

<h2>Per-case detail</h2>
$case_blocks
</body>
</html>
"""
)

_CASE = Template(
    """<div class="case">
  <p class="q">$question</p>
  <p><em>Answer:</em> $answer</p>
  $metric_blocks
</div>"""
)

_METRIC = Template(
    """<div class="metric">
    <p><strong>$metric:</strong> $score</p>
    <p class="rationale">$rationale</p>
    $raw_block
  </div>"""
)


def _fmt(score: float) -> str:
    return f"{score:.3f}"


def render_html(case_reports: list[CaseReport], agg: Aggregate) -> str:
    metric_rows = "\n".join(
        f"<tr><td>{html.escape(m)}</td><td>{_fmt(s)}</td></tr>"
        for m, s in agg.per_metric.items()
    )
    overall = _fmt(agg.overall) if agg.overall is not None else "n/a"

    case_blocks = []
    for cr in case_reports:
        metric_blocks = []
        for r in cr.results:
            raw_block = ""
            if r.raw_judge_output is not None:
                raw_block = (
                    "<details><summary>raw judge output</summary>"
                    f"<pre>{html.escape(r.raw_judge_output)}</pre></details>"
                )
            metric_blocks.append(
                _METRIC.substitute(
                    metric=html.escape(r.metric),
                    score=_fmt(r.score),
                    rationale=html.escape(r.rationale),
                    raw_block=raw_block,
                )
            )
        case_blocks.append(
            _CASE.substitute(
                question=html.escape(cr.case.question),
                answer=html.escape(cr.case.generated_answer),
                metric_blocks="\n".join(metric_blocks),
            )
        )

    return _PAGE.substitute(
        n_cases=agg.n_cases,
        metric_rows=metric_rows,
        overall=overall,
        case_blocks="\n".join(case_blocks),
    )
