from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    fake: bool = False
    share: bool = False
    host: str = "127.0.0.1"
    port: int = 7860
    silence: float = 0.7

    @classmethod
    def load(cls):
        values = {**dotenv_values(os.getenv("ASR_ENV_PATH", str(ROOT / ".env"))), **os.environ}
        def get(key, default):
            return values.get(key) or default
        settings = cls(
            fake=get("WEB_FAKE", "false").lower() in {"1", "true", "yes"},
            share=get("WEB_SHARE", "false").lower() in {"1", "true", "yes"},
            host=get("WEB_HOST", "127.0.0.1"),
            port=int(get("WEB_PORT", "7860")),
            silence=float(get("SEGMENT_SILENCE_SECONDS", "0.7")),
        )
        if not 0 < settings.silence < 30 or not 1 <= settings.port <= 65535:
            raise ValueError("Silence/port không hợp lệ")
        return settings
