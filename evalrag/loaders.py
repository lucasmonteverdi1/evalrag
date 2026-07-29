from __future__ import annotations

import importlib
import json
from pathlib import Path

import yaml

from evalrag.config import _resolve_config
from evalrag.runner.adapter import PipelineAdapter
from evalrag.runner.loop import EvalInput
from evalrag.types import Chunk


class LoaderError(ValueError):
    """Raised when a CLI-supplied resource can't be loaded."""


def load_adapter(spec: str) -> PipelineAdapter:
    """Import a PipelineAdapter from a 'package.module:attribute' spec.

    The attribute may be an adapter instance or a zero-arg callable/class that
    returns one. This is the extensibility seam: the user plugs in their pipeline
    without evalrag knowing how it's built.

    SECURITY: importing the module runs its top-level code, so `spec` must come from
    a trusted source (a CLI operator or CI config), like `pytest --plugin` or
    `gunicorn module:app`. Never pass an adapter spec derived from untrusted input.
    """
    if ":" not in spec:
        raise LoaderError(f"adapter spec must be 'module:attribute', got {spec!r}")
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise LoaderError(f"could not import adapter module {module_name!r}: {e}") from e
    try:
        obj = getattr(module, attr)
    except AttributeError as e:
        raise LoaderError(f"{module_name!r} has no attribute {attr!r}") from e
    adapter = obj() if callable(obj) else obj
    if not hasattr(adapter, "run"):
        raise LoaderError(f"adapter {spec!r} has no .run(question) method")
    return adapter


def load_thresholds(path: str | Path = "configs/thresholds.yaml") -> dict[str, float]:
    """Load per-metric thresholds, dropping null (ungated) and non-numeric entries."""
    raw = yaml.safe_load(_resolve_config(path, "thresholds.yaml").read_text()) or {}
    thresholds: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            thresholds[key] = float(value)
    return thresholds


def load_inputs(path: str | Path) -> list[EvalInput]:
    """Load EvalInputs from a JSON file: a list of {question, expected_answer?,
    source_chunk_id?} objects."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise LoaderError("inputs file must be a JSON list of objects")
    inputs: list[EvalInput] = []
    for i, item in enumerate(data):
        if "question" not in item:
            raise LoaderError(f"inputs[{i}] missing required 'question'")
        inputs.append(
            EvalInput(
                question=item["question"],
                expected_answer=item.get("expected_answer"),
                source_chunk_id=item.get("source_chunk_id"),
            )
        )
    return inputs


def load_documents(path: str | Path) -> list[Chunk]:
    """Load source documents (chunks) for generation from a JSON file:
    a list of {id, text} objects."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise LoaderError("documents file must be a JSON list of {id, text} objects")
    return [Chunk(item["id"], item["text"]) for item in data]
