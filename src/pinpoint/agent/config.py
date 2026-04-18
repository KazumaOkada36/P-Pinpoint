"""Config management — stored at ~/.pinpoint/config.json"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".pinpoint"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history"
MODEL_FILE = CONFIG_DIR / "recommender.pkl"
CUSTOM_DATA_DIR = CONFIG_DIR / "custom_data"

DEFAULTS = {
    "api_key": "",
    "model": "claude-sonnet-4-6",
    "top_n": 5,
    "data_dir": "",  # empty = use bundled Data_Set/clean_data
}

MODELS = {
    "claude-opus-4-6": "Opus 4.6  — most capable",
    "claude-sonnet-4-6": "Sonnet 4.6 — balanced",
    "claude-haiku-4-5-20251001": "Haiku 4.5  — fastest",
}


class Config:
    def __init__(self, data: dict) -> None:
        self._data = {**DEFAULTS, **data}

    # ------------------------------------------------------------------
    @classmethod
    def load(cls) -> "Config":
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
            except Exception:
                data = {}
        else:
            data = {}
        # env var override
        env_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("PINPOINT_API_KEY")
        if env_key:
            data["api_key"] = env_key
        return cls(data)

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(self._data, indent=2))

    # ------------------------------------------------------------------
    @property
    def api_key(self) -> str:
        return self._data.get("api_key", "")

    @api_key.setter
    def api_key(self, v: str) -> None:
        self._data["api_key"] = v

    @property
    def model(self) -> str:
        return self._data.get("model", DEFAULTS["model"])

    @model.setter
    def model(self, v: str) -> None:
        self._data["model"] = v

    @property
    def top_n(self) -> int:
        return int(self._data.get("top_n", DEFAULTS["top_n"]))

    @top_n.setter
    def top_n(self, v: int) -> None:
        self._data["top_n"] = v

    @property
    def data_dir(self) -> Optional[Path]:
        d = self._data.get("data_dir", "")
        return Path(d) if d else None

    @data_dir.setter
    def data_dir(self, v: str) -> None:
        self._data["data_dir"] = v

    @property
    def masked_key(self) -> str:
        k = self.api_key
        if not k:
            return "(not set)"
        if len(k) <= 8:
            return "***"
        return f"{k[:7]}...{k[-4:]}"

    def as_dict(self) -> dict:
        return dict(self._data)
