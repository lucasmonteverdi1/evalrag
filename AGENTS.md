# EvalRAG — Agent Context

## What this project is

EvalRAG is a CLI evaluation framework for **RAG pipelines**, built **from scratch**
without wrapping RAGAS, TruLens, DeepEval, or similar. It answers: "how do you know
if your RAG pipeline is actually working well?" by producing automated, measurable,
reproducible quality scores.

This is a **portfolio project**. Architecture clarity, traceable scoring decisions,
clean commit history, and tests matter as much as raw functionality. The whole point
is to demonstrate AI-engineering depth beyond LLM API wrappers, so avoid magic and
make every scoring decision explainable.

## Design goals (non-negotiable)

- **No magic** — every score is traceable to an explicit, inspectable decision.
- **Reproducibility** — same dataset + same pipeline = same scores.
- **Extensibility** — users plug in custom scorers and pipeline adapters.
- **CI-friendly** — exits non-zero when aggregate scores fall below thresholds.

## Tech stack & layout

- **Language: Python** (3.11+). Use `uv` for env/deps.
- **Library-first**: all logic lives in the importable `evalrag/` package; the CLI is
  a thin wrapper that parses args and calls the library. (Think service layer vs.
  controller — no business logic in the CLI.)

```
evalrag/
├── pyproject.toml
├── README.md
├── AGENTS.md                # canonical agent context (CLAUDE.md re-exports this)
├── CLAUDE.md                # stub: @AGENTS.md
├── evalrag/                 # importable core library
│   ├── __init__.py
│   ├── types.py             # EvalCase, MetricResult, Chunk (dataclasses)
│   ├── config.py            # frozen config dataclasses loaded from configs/*.yaml
│   ├── generator/           # synthetic (question, expected_answer) generation
│   ├── runner/              # pipeline adapter + execution loop
│   ├── scorer/              # metric implementations (faithfulness.py done)
│   ├── judge/               # LLM-as-judge client + versioned prompt loading
│   ├── validation/          # offline judge-validation harness
│   ├── report/              # JSON + HTML rendering
│   └── cli.py               # thin CLI entrypoint (exposed via pyproject script)
├── prompts/                 # versioned judge prompts (e.g. faithfulness_v1.md)
├── configs/                 # YAML: models, thresholds, prompt versions
├── testdata/                # importable pkg: MetricStub + hand-written stub eval cases
├── tests/
└── docs/                    # architecture decisions, metric definitions
```

## Core data types (define these first)

```python
@dataclass(frozen=True)
class Chunk:
    id: str
    text: str

@dataclass(frozen=True)
class EvalCase:
    question: str
    generated_answer: str            # from the runner
    retrieved_chunks: list[Chunk]    # from the runner
    expected_answer: str | None = None     # from generator; only reference-based metrics use it
    source_chunk_id: str | None = None     # ground-truth relevant chunk, if known

@dataclass(frozen=True)
class MetricResult:
    metric: str
    score: float                     # normalized 0.0–1.0
    rationale: str                   # human-readable explanation (traceability)
    raw_judge_output: str | None = None    # stored verbatim for reproducibility
```

## Pipeline adapter contract (the key extensibility seam)

The system under test is a **pipeline, not a single LLM** — a retriever + a generator,
treated as a black box. It is accessed ONLY through this contract:

```python
class PipelineAdapter(Protocol):
    def run(self, question: str) -> tuple[list[Chunk], str]:
        """Return (retrieved_chunks, generated_answer) for a question."""
```

The tool must not assume how the pipeline is built internally (LangChain, LlamaIndex,
raw code — irrelevant). Define this interface before anything that depends on it.

## The three metrics — each reads DIFFERENT inputs

This is the most important correctness rule. The metrics are **not** all "similarity to
the expected answer." Implement them as distinct relationships:

- **Faithfulness** — inputs: `generated_answer` + `retrieved_chunks`. Decompose the
  answer into atomic claims; check each claim is grounded in the retrieved context.
  Score = fraction of claims supported. **Reference-free** (does NOT use expected_answer).
  LLM-as-judge.
- **Answer relevance** — inputs: `generated_answer` + `question`. Does the answer
  address the question? Judge, optionally backed by embedding similarity. Reference-free.
- **Context precision** — inputs: `retrieved_chunks` + (`source_chunk_id` if known, else
  `question`). Of the chunks retrieved, how many were relevant, and were relevant ones
  ranked high (rank-aware)? If ground-truth chunk IDs exist (the generator records them),
  compare IDs deterministically; otherwise judge per-chunk relevance.

Optional later metrics (not part of the MVP): **answer correctness** (the one that DOES
compare `generated_answer` to `expected_answer`) and **context recall**.

## LLM-as-judge conventions

- Judge prompts live in `prompts/` as **versioned files** (e.g. `faithfulness_v1.md`).
  Load by version; never inline prompts in code.
- The **judge model must be configurable and should differ from the pipeline's
  generator model** (self-preference bias inflates scores when they match).
- Judge calls use **temperature 0**. Store the **raw judge output** and a **rationale**
  on every `MetricResult` for traceability.
- Ask the judge for chain-of-thought + a structured verdict; parse the verdict robustly.

## Judge validation (the differentiator — OFFLINE, not per-run)

Validation calibrates the *judge itself*; it is NOT a step in the per-question scoring
loop and does NOT affect the evaluation run's exit code.

- Hand-label a small holdout (~50–100 cases) for a metric.
- Run the judge on the same cases.
- Compute agreement: correlation (Spearman/Kendall) for continuous, agreement %/F1 for
  categorical.
- High agreement → the judge is trustworthy. Low → fix the prompt / swap the judge model /
  add bias mitigation, then re-validate. This lives in `evalrag/validation/`.

## Reproducibility rules

- Temperature 0 everywhere; pin and record model version strings.
- Cache every LLM/judge response keyed by (model, prompt_version, inputs); re-runs read
  the cache. Persist raw responses.

## Build order (scorer-first phasing) — status

Build against the data flow in REVERSE so the novel part is proven first.

- [x] **1.** `types.py` — `EvalCase`, `MetricResult`, `Chunk`.
- [x] **2.** `judge/` — judge client (configurable model, temp 0, versioned prompt
  loading, response caching, raw-output capture) + `config.py`.
- [x] **3.** `scorer/faithfulness.py` — ONE metric, end to end, over hand-written stub
  `EvalCase`s in `testdata/`; prints per-case scores + rationales via an offline demo.
- [ ] **4.** `validation/` — label those stubs by hand; compute judge↔human agreement;
  confirm the judge is trustworthy. ← **next**
- [ ] **5.** THEN: `runner/` (adapter + loop) → `scorer/` relevance + precision →
  `generator/` → `report/` (JSON + HTML) → CLI threshold-gating + exit codes.

## Repo conventions (decisions already made — keep consistent)

- **Env/deps:** `uv` only. Run via `uv run` (`uv run pytest`, `uv run ruff check .`,
  `uv run python -m evalrag.scorer` for the faithfulness demo).
- **Provider abstraction (two layers):** scorers/judge depend only on the `LLMProvider`
  Protocol (`evalrag/judge/provider.py`); never import a vendor SDK directly. The real
  transport is the **OpenAI SDK pointed at OpenRouter** via a configurable `base_url`, so
  model strings are vendor-qualified (`anthropic/...`, `openai/...`, `google/...`).
- **Config (`evalrag/config.py`):** frozen dataclasses loaded from `configs/*.yaml` with
  `pyyaml`. Precedence is **CLI flag > env var > YAML default** (applied in
  `load_models_config` via the `overrides` dict). **Secrets come from env only** — YAML
  names *which* env var holds the key (`api_key_env`), never the value.
- **Self-preference guard:** `judge_matches_generator()` is the pure predicate; the CLI
  (when built) warns + prompts to continue when judge and generator models match.
- **Testing is hermetic:** tests and the demo use the deterministic `FakeProvider` (no
  network, no key). The real OpenRouter path is exercised only manually with a key — CI
  never hits the network.
- **Test data is a package:** `testdata/` is importable; stubs use the generic
  `MetricStub[L]` (`testdata/stubs.py`) so every metric reuses one shape. Stub ground-truth
  labels generate the FakeProvider verdicts, so they can't drift from the claims.
- **CI / hooks:** `.github/workflows/ci.yml` runs ruff + pytest on push/PR to main;
  `.pre-commit-config.yaml` runs them before each commit.
- **Commits:** small and well-described; settings/config commits kept separate from
  feature commits.
