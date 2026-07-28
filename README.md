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

## Design

See [AGENTS.md](AGENTS.md) for the full architecture, metric definitions, and the
scorer-first build order.
