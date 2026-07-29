# EvalRAG

A from-scratch CLI + library for evaluating **RAG pipelines** — no RAGAS, TruLens, or
DeepEval wrappers.

Answers *"how do I know my RAG pipeline is actually working well?"* by producing
automated, reproducible quality scores across three metrics:

- **Faithfulness** — is every claim in the answer grounded in the retrieved chunks?
- **Answer relevance** — does the answer actually address the question?
- **Context precision** — did the retriever surface relevant chunks, ranked high?

Every score is traceable to an explicit, inspectable decision (the raw judge output is
kept). It exits non-zero when a metric falls below its threshold, so it works as a **CI
gate**.

## Install

```bash
# From GitHub (no clone needed):
pip install git+https://github.com/lucasmonteverdi1/evalrag.git

# Or with uv, from a clone:
uv sync --all-groups
```

You need an [OpenRouter](https://openrouter.ai) API key (one key, any vendor's models):

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

## Evaluate your RAG pipeline

### 1. Write an adapter

EvalRAG treats your pipeline as a black box. Implement one method — `run(question)` that
returns `(retrieved_chunks, generated_answer)`:

```python
# my_eval.py
from evalrag.types import Chunk

class MyRAGAdapter:
    def run(self, question: str) -> tuple[list[Chunk], str]:
        # Whatever your pipeline is (LangChain, LlamaIndex, raw code — doesn't matter):
        docs = my_retriever.search(question)
        answer = my_llm.generate(question, docs)
        return [Chunk(id=d.id, text=d.text) for d in docs], answer

adapter = MyRAGAdapter()   # module-level instance evalrag will import
```

### 2. Provide questions

A JSON list of questions to evaluate. `expected_answer` / `source_chunk_id` are optional
(the latter enables context precision's deterministic, no-LLM path):

```json
[
  {"question": "What is the capital of France?", "source_chunk_id": "doc-42"},
  {"question": "What is the return policy?"}
]
```

### 3. Run

```bash
evalrag --adapter my_eval:adapter --inputs questions.json --out-dir eval-out
```

This runs your pipeline over each question, scores all three metrics, writes
`eval-out/report.json` + `eval-out/report.html`, and exits **0** if every metric is at
or above its threshold, **1** if any falls below, **2** on a config/usage error.

Don't have questions yet? Generate them from your documents instead of `--inputs`:

```bash
evalrag --adapter my_eval:adapter --generate documents.json
# documents.json: [{"id": "doc-42", "text": "..."}, ...]
```

Try it with no code using the bundled demo adapter:

```bash
echo '[{"question":"What is the capital of France?"}]' > q.json
evalrag --adapter evalrag.demo_adapter:demo_adapter --inputs q.json
```

## Reading the reports

Each run writes two files to `--out-dir`:

**`report.html`** — open in a browser. Top section is a **per-metric score table** plus
an *overall* number (informational only — gating is per-metric, not on the overall).
Below that, each evaluated case shows its question, the pipeline's answer, and every
metric's score + rationale. Expand **"raw judge output"** on any metric to see the
judge's verbatim reasoning — this is the traceability guarantee: no score is a black box.

**`report.json`** — the same data, machine-readable, for dashboards or diffing across
runs:

```jsonc
{
  "summary": {
    "n_cases": 2,
    "per_metric": { "faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 1.0 },
    "overall": 1.0            // informational, NOT used for gating
  },
  "cases": [
    {
      "question": "...",
      "generated_answer": "...",
      "metrics": [
        { "metric": "faithfulness", "score": 1.0, "rationale": "1/1 claims grounded",
          "raw_judge_output": "..." }   // the judge's full response, kept verbatim
      ]
    }
  ]
}
```

**How to read the numbers:** each metric is 0.0–1.0, higher is better. A metric **fails**
(and the run exits non-zero) when its mean score is below the threshold in
`configs/thresholds.yaml`. The terminal prints a `[PASS]`/`[FAIL]` line per metric and a
final `Result: PASS`/`FAIL`.

## Use in CI

`evalrag`'s exit code gates the build:

```yaml
# .github/workflows/eval.yml
- run: pip install git+https://github.com/lucasmonteverdi1/evalrag.git
- run: evalrag --adapter my_eval:adapter --inputs questions.json
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Thresholds live in `configs/thresholds.yaml` (override with `--thresholds-config`).

## Use as a library

Every metric is importable and takes `(EvalCase, judge)`:

```python
from evalrag.scorer.faithfulness import score_faithfulness
result = score_faithfulness(eval_case, judge)   # -> MetricResult(score, rationale, ...)
```

## Configuration

- `configs/models.yaml` — judge and system-under-test models (the judge **must** differ
  from the model being evaluated, to avoid self-preference bias). Secrets come from env
  vars named here (`api_key_env`), never stored in YAML.
- `configs/thresholds.yaml` — per-metric pass thresholds.
- `configs/prompts.yaml` — pinned judge-prompt versions.

## Troubleshooting

**`ModuleNotFoundError: No module named 'evalrag.cli'` during local development.**
Only affects the editable dev install (`uv sync`), never a `pip install` of the package.
It happens if you mix `uv pip install/uninstall` with `uv sync`. Reset the environment:

```bash
rm -rf .venv && uv sync --all-groups
```

Then use `uv run evalrag ...`. (Don't mix `uv pip` and `uv sync` in the same venv.)

## Design

See [AGENTS.md](AGENTS.md) for the full architecture, metric definitions, and the
scorer-first build order.
