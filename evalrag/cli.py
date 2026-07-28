from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evalrag.config import (
    ConfigError,
    judge_matches_sut,
    load_models_config,
    load_prompt_versions,
    resolve_api_key,
)
from evalrag.gating import GateResult, evaluate_thresholds
from evalrag.generator.synthesize import generate_inputs
from evalrag.judge.cache import ResponseCache
from evalrag.judge.llm_client import LLMClient
from evalrag.judge.openrouter_provider import OpenRouterProvider
from evalrag.loaders import (
    LoaderError,
    load_adapter,
    load_documents,
    load_inputs,
    load_thresholds,
)
from evalrag.report.aggregate import aggregate
from evalrag.report.html_report import render_html
from evalrag.report.json_report import render_json
from evalrag.runner.loop import run_pipeline
from evalrag.scorer.run_all import score_run


def _build_client(provider_cfg, cache, prompt_versions) -> LLMClient:
    """Build an LLMClient backed by the real OpenRouter provider."""
    api_key = resolve_api_key(provider_cfg)
    provider = OpenRouterProvider(
        api_key=api_key,
        base_url=provider_cfg.base_url,
        max_retries=provider_cfg.max_retries,
    )
    return LLMClient(
        provider=provider,
        config=provider_cfg,
        cache=cache,
        prompt_versions=prompt_versions,
    )


def _confirm_self_preference(skip: bool) -> bool:
    """Warn + prompt when judge and SUT are the same model. Returns True to proceed."""
    msg = (
        "WARNING: the judge and the system-under-test use the same model. Scores may "
        "be inflated by self-preference bias."
    )
    print(msg, file=sys.stderr)
    if skip or not sys.stdin.isatty():
        return True  # non-interactive / --yes: proceed but the warning was shown
    answer = input("Continue anyway? [Y/n] ").strip().lower()
    return answer in ("", "y", "yes")


def _print_gate(gate: GateResult) -> None:
    print("\n=== Threshold gating ===")
    for g in gate.gates:
        status = "PASS" if g.passed else "FAIL"
        print(f"  [{status}] {g.metric}: {g.score:.3f} (threshold {g.threshold:.2f})")
    print(f"\nResult: {'PASS' if gate.passed else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalrag", description="Evaluate a RAG pipeline.")
    parser.add_argument("--adapter", required=True,
                        help="PipelineAdapter as 'module:attribute'")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", help="JSON file of EvalInputs to evaluate")
    src.add_argument("--generate", help="JSON file of documents to generate questions from")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--thresholds-config", default="configs/thresholds.yaml")
    parser.add_argument("--out-dir", default="eval-out", help="where to write reports")
    parser.add_argument("--n-per-chunk", type=int, default=1)
    parser.add_argument("-y", "--yes", action="store_true",
                        help="skip the self-preference confirmation prompt")
    args = parser.parse_args(argv)

    try:
        models = load_models_config(args.models_config)
        prompt_versions = load_prompt_versions()
        thresholds = load_thresholds(args.thresholds_config)
        adapter = load_adapter(args.adapter)
    except (ConfigError, LoaderError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Self-preference guard (judge vs system-under-test).
    if judge_matches_sut(models) and not _confirm_self_preference(args.yes):
        print("Aborted.", file=sys.stderr)
        return 2

    cache = ResponseCache(enabled=models.cache.enabled, cache_dir=models.cache.dir)

    try:
        # 1. Get the eval inputs (generate from docs, or load a dataset).
        if args.generate:
            generator = _build_client(models.system_under_test, cache, prompt_versions)
            documents = load_documents(args.generate)
            inputs = generate_inputs(documents, generator, n_per_chunk=args.n_per_chunk)
        else:
            inputs = load_inputs(args.inputs)

        # 2. Run the pipeline -> EvalCases.
        run = run_pipeline(inputs, adapter)

        # 3. Score every case with the judge.
        judge = _build_client(models.judge, cache, prompt_versions)
        case_reports = score_run(run, judge)
    except (ConfigError, LoaderError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # 4. Aggregate + write reports.
    agg = aggregate(case_reports)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(render_json(case_reports, agg))
    (out_dir / "report.html").write_text(render_html(case_reports, agg))
    print(f"Wrote reports to {out_dir}/ ({agg.n_cases} cases, {len(run.errors)} errors)")

    # 5. Threshold gating -> exit code. This is the CI-gating promise.
    gate = evaluate_thresholds(agg, thresholds)
    _print_gate(gate)
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())