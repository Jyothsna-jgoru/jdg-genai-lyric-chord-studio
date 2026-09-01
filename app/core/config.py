from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "storage" / "model-cache"))


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "JDG GenAI Lyric-to-Chord Studio"
    api_prefix: str = "/api/v1"
    database_url: str = os.getenv("JDG_DATABASE_URL", f"sqlite:///{ROOT / 'storage' / 'jdg_studio.db'}")
    base_model: str = os.getenv("JDG_BASE_MODEL", "google/flan-t5-small")
    adapter_path: Path = Path(os.getenv("JDG_ADAPTER_PATH", str(ROOT / "storage" / "adapters" / "dev")))
    max_lyric_chars: int = int(os.getenv("JDG_MAX_LYRIC_CHARS", "20000"))
    model_autoload: bool = _boolean("JDG_MODEL_AUTOLOAD", True)
    allow_model_download: bool = _boolean("JDG_ALLOW_MODEL_DOWNLOAD", False)
    log_level: str = os.getenv("JDG_LOG_LEVEL", "INFO")
    exports_dir: Path = ROOT / "storage" / "exports"
    evaluation_path: Path = ROOT / "storage" / "evaluation_results.json"


settings = Settings()
