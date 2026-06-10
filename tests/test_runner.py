from evalrag.judge.cache import ResponseCache
from evalrag.judge.client import JudgeClient
from evalrag.judge.provider import FakeProvider
from evalrag.config import ProviderConfig
from evalrag.runner.adapter import FakePipelineAdapter
from evalrag.runner.loop import EvalInput, RunError, RunResult, run_pipeline
from evalrag.scorer.faithfulness import score_faithfulness
from evalrag.types import Chunk


def make_adapter(responses=None, raises=None) -> FakePipelineAdapter:
    return FakePipelineAdapter(responses=responses or {}, raises=raises or {})


class TestRunPipeline:
    def test_happy_path_builds_cases(self):
        responses = {
            "q1": ([Chunk("c1", "ctx one")], "answer one"),
            "q2": ([Chunk("c2", "ctx two")], "answer two"),
        }
        inputs = [
            EvalInput("q1", expected_answer="exp1", source_chunk_id="c1"),
            EvalInput("q2"),
        ]
        result = run_pipeline(inputs, make_adapter(responses))
        assert isinstance(result, RunResult)
        assert len(result.cases) == 2
        assert result.errors == []
        # First case carries ground truth through from the input.
        c1 = result.cases[0]
        assert c1.question == "q1"
        assert c1.generated_answer == "answer one"
        assert c1.expected_answer == "exp1"
        assert c1.source_chunk_id == "c1"
        # Second case has no ground truth.
        assert result.cases[1].expected_answer is None

    def test_chunks_land_as_tuple(self):
        # Adapter returns a list[Chunk]; EvalCase.__post_init__ coerces to tuple.
        responses = {"q": ([Chunk("c1", "x")], "a")}
        result = run_pipeline([EvalInput("q")], make_adapter(responses))
        assert isinstance(result.cases[0].retrieved_chunks, tuple)

    def test_error_is_captured_and_run_continues(self):
        responses = {"ok": ([Chunk("c1", "ctx")], "good answer")}
        raises = {"boom": RuntimeError("pipeline exploded")}
        inputs = [EvalInput("boom"), EvalInput("ok")]
        result = run_pipeline(inputs, make_adapter(responses, raises))
        # The good input still produced a case.
        assert len(result.cases) == 1
        assert result.cases[0].question == "ok"
        # The bad input is captured, not raised.
        assert len(result.errors) == 1
        assert result.errors[0] == RunError(question="boom", error="pipeline exploded")

    def test_empty_inputs_no_calls(self):
        adapter = make_adapter()
        result = run_pipeline([], adapter)
        assert result.cases == []
        assert result.errors == []
        assert adapter.calls == 0

    def test_unknown_question_becomes_error(self):
        # FakePipelineAdapter raises KeyError for questions with no canned response.
        result = run_pipeline([EvalInput("unmapped")], make_adapter())
        assert result.cases == []
        assert len(result.errors) == 1
        assert result.errors[0].question == "unmapped"


class TestGlueWithScorer:
    def test_produced_case_is_score_ready(self, tmp_path):
        # Prove a runner-built EvalCase flows straight into score_faithfulness.
        responses = {
            "q": ([Chunk("c1", "Paris is the capital of France.")],
                  "Paris is the capital of France."),
        }
        result = run_pipeline([EvalInput("q")], make_adapter(responses))
        case = result.cases[0]

        verdict = '```json\n{"claims": [{"index": 0, "supported": true}]}\n```'
        judge = JudgeClient(
            provider=FakeProvider(responses={"Paris": verdict}),
            config=ProviderConfig(
                provider="fake",
                base_url="n/a",
                api_key_env="UNUSED",
                model="fake-model",
                temperature=0,
                max_tokens=1024,
            ),
            cache=ResponseCache(enabled=False, cache_dir=tmp_path),
            prompt_versions={"faithfulness": "v1"},
        )
        metric = score_faithfulness(case, judge)
        assert metric.metric == "faithfulness"
        assert metric.score == 1.0
