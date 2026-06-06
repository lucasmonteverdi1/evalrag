from __future__ import annotations

from pathlib import Path


class PromptNotFoundError(FileNotFoundError):
    """Raised when a versioned prompt file does not exist."""


def load_prompt(
    metric: str,
    version: str,
    prompts_dir: str | Path = "prompts",
) -> str:
    path = Path(prompts_dir) / f"{metric}_{version}.md"
    if not path.is_file():
        raise PromptNotFoundError(f"prompt not found: {path}")
    return path.read_text()