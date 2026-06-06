from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ResponseCache:
    def __init__(self, enabled: bool, cache_dir: str | Path) -> None:
        self.enabled = enabled
        self.dir = Path(cache_dir)

    @staticmethod
    def make_key(provider: str, model: str, prompt_version: str, prompt_text: str) -> str:
        h = hashlib.sha256()
        # Join with a separator that can't appear ambiguously across fields.
        payload = "\x1f".join([provider, model, prompt_version, prompt_text])
        h.update(payload.encode("utf-8"))
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        return json.loads(path.read_text())["response"]

    def set(self, key: str, response: str, *, meta: dict | None = None) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {"response": response, "meta": meta or {}}
        self._path(key).write_text(json.dumps(record, indent=2))