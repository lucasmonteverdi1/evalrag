import json

import pytest

from evalrag import cli
from evalrag.gating import evaluate_thresholds
from evalrag.loaders import (
    LoaderError,
    load_adapter,
    load_inputs,
    load_thresholds,
)
from evalrag.report.aggregate import aggregate, CaseReport
from evalrag.types import Chunk, EvalCase, MetricResult


# --- gating ---
def _agg(**scores):
    reports = [
        CaseReport(
            case=EvalCase("q", "a", (Chunk("c", "x"),)),
            results=[MetricResult(m, s, "r") for m, s in scores.items()],
        )
    ]
    return aggregate(reports)


def test_gate_passes_when_all_above():
    gate = evaluate_thresholds(_agg(faithfulness=0.9), {"faithfulness": 0.8})
    assert gate.passed


def test_gate_fails_when_one_below():
    gate = evaluate_thresholds(
        _agg(faithfulness=0.5, answer_relevance=0.9),
        {"faithfulness": 0.8, "answer_relevance": 0.7},
    )
    assert not gate.passed
    assert [g.metric for g in gate.gates if not g.passed] == ["faithfulness"]


def test_gate_ignores_ungated_metric():
    # No threshold for context_precision -> not gated, run still passes.
    gate = evaluate_thresholds(_agg(context_precision=0.1), {"faithfulness": 0.8})
    assert gate.passed
    assert gate.gates == []


# --- loaders ---
def test_load_thresholds_drops_null_and_bools(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("faithfulness: 0.8\nanswer_correctness: null\nstrict_mode: false\n")
    assert load_thresholds(p) == {"faithfulness": 0.8}


def test_load_adapter_dotted_path():
    adapter = load_adapter("evalrag.demo_adapter:demo_adapter")
    assert hasattr(adapter, "run")


def test_load_adapter_bad_spec():
    with pytest.raises(LoaderError):
        load_adapter("no_colon_here")


def test_load_inputs(tmp_path):
    p = tmp_path / "i.json"
    p.write_text(json.dumps([{"question": "Q1", "source_chunk_id": "c1"}]))
    inputs = load_inputs(p)
    assert inputs[0].question == "Q1"
    assert inputs[0].source_chunk_id == "c1"


def test_load_inputs_missing_question(tmp_path):
    p = tmp_path / "i.json"
    p.write_text(json.dumps([{"expected_answer": "x"}]))
    with pytest.raises(LoaderError):
        load_inputs(p)


# --- CLI end-to-end (hermetic: patch the LLMClient builder to use FakeProvider) ---
def _fake_client_builder(monkeypatch):
    from evalrag.judge.cache import ResponseCache
    from evalrag.judge.llm_client import LLMClient
    from evalrag.judge.provider import FakeProvider
    from evalrag.config import ProviderConfig

    # Canned verdicts that make the demo-adapter answers score high.
    responses = {
        "Paris": '```json\n{"claims": [{"index": 0, "supported": true}], '
        '"score": 1.0, "reason": "ok", "chunks": [{"index": 0, "relevant": true}]}\n```',
    }

    def builder(provider_cfg, cache, prompt_versions):
        return LLMClient(
            provider=FakeProvider(
                responses=responses,
                default='```json\n{"claims": [], "score": 1.0, "chunks": []}\n```',
            ),
            config=ProviderConfig("fake", "https://x", "K", "m", 0, 512),
            cache=ResponseCache(enabled=False, cache_dir="/tmp/evalrag-test-cache"),
            prompt_versions=prompt_versions,
        )

    monkeypatch.setattr(cli, "_build_client", builder)


def test_cli_end_to_end_writes_reports_and_gates(tmp_path, monkeypatch):
    _fake_client_builder(monkeypatch)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps([{"question": "What is the capital of France?"}]))
    out = tmp_path / "out"

    code = cli.main(
        [
            "--adapter", "evalrag.demo_adapter:demo_adapter",
            "--inputs", str(inputs),
            "--out-dir", str(out),
            "--thresholds-config", "configs/thresholds.yaml",
        ]
    )
    assert code in (0, 1)  # ran gating; exit reflects pass/fail
    assert (out / "report.json").is_file()
    assert (out / "report.html").is_file()


def test_cli_bad_adapter_returns_2(tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text("[]")
    code = cli.main(
        ["--adapter", "does.not:exist", "--inputs", str(inputs), "--out-dir", str(tmp_path / "o")]
    )
    assert code == 2
