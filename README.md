# EvalRAG

A from-scratch CLI evaluation framework for RAG pipelines — no RAGAS, TruLens, or DeepEval wrappers.

Answers: "how do you know if your RAG pipeline is actually working well?" by producing automated, measurable, reproducible quality scores across three metrics: **faithfulness**, **answer relevance**, and **context precision**.

## Quickstart

```bash
# Install uv (if not already installed)
curl -Ls https://astral.sh/uv/install.sh | sh

# Install the project and dev deps
uv sync --all-groups

# Run tests
uv run pytest

# Run the CLI (coming soon)
uv run evalrag --help
```

## Design

Every score is traceable to an explicit, inspectable decision. See [CLAUDE.md](CLAUDE.md) for the full architecture, metric definitions, and build order.
