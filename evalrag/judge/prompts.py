from __future__ import annotations

from pathlib import Path

# Prompts ship inside the package, so the default resolves relative to the package
# (works when pip-installed), not the current working directory.
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptNotFoundError(FileNotFoundError):
    """Raised when a versioned prompt file does not exist."""


def load_prompt(
    metric: str,
    version: str,
    prompts_dir: str | Path | None = None,
) -> str:
    base = Path(prompts_dir) if prompts_dir is not None else _DEFAULT_PROMPTS_DIR
    path = base / f"{metric}_{version}.md"
    if not path.is_file():
        raise PromptNotFoundError(f"prompt not found: {path}")
    return path.read_text()
