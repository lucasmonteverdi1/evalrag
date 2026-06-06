import pytest

from evalrag.judge.prompts import PromptNotFoundError, load_prompt


def test_loads_existing_prompt(tmp_path):
    (tmp_path / "faithfulness_v1.md").write_text("JUDGE PROMPT BODY")
    text = load_prompt("faithfulness", "v1", prompts_dir=tmp_path)
    assert text == "JUDGE PROMPT BODY"


def test_filename_is_metric_underscore_version(tmp_path):
    (tmp_path / "answer_relevance_v2.md").write_text("body")
    assert load_prompt("answer_relevance", "v2", prompts_dir=tmp_path) == "body"


def test_missing_prompt_raises(tmp_path):
    with pytest.raises(PromptNotFoundError, match="faithfulness_v9.md"):
        load_prompt("faithfulness", "v9", prompts_dir=tmp_path)
