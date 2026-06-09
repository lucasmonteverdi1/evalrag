from __future__ import annotations

import sys

from evalrag.config import (
    ConfigError,
    load_models_config,
    load_prompt_versions,
    resolve_api_key,
)
from evalrag.judge.cache import ResponseCache
from evalrag.judge.client import JudgeClient
from evalrag.judge.openrouter_provider import OpenRouterProvider
from evalrag.validation.harness import AgreementReport, validate_faithfulness
from testdata.faithfulness_stubs import STUBS


def _print_report(report: AgreementReport) -> None:
    c = report.confusion
    print("\n=== Faithfulness judge validation ===")
    print(f"claims evaluated: {report.n_claims}")
    print(f"agreement:        {report.agreement_pct:.1%}")
    print(f"F1:               {report.f1:.3f}")
    print(f"Cohen's kappa:    {report.kappa:.3f}")
    print(f"confusion:        tp={c.tp} fp={c.fp} tn={c.tn} fn={c.fn}")
    print("\nper stub:")
    for s in report.per_stub:
        print(f"  - {s.question}")
        print(f"      human: {s.human}")
        print(f"      judge: {s.judge}")


def main() -> int:
    models = load_models_config()
    try:
        api_key = resolve_api_key(models.judge)
    except ConfigError as e:
        print(f"{e}\nValidation needs the real judge; skipping.", file=sys.stderr)
        return 0

    provider = OpenRouterProvider(
        api_key=api_key,
        base_url=models.judge.base_url,
        max_retries=models.judge.max_retries,
    )
    judge = JudgeClient(
        provider=provider,
        config=models.judge,
        cache=ResponseCache(enabled=models.cache.enabled, cache_dir=models.cache.dir),
        prompt_versions=load_prompt_versions(),
    )

    report = validate_faithfulness(STUBS, judge)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
