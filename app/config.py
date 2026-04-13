from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    auth_password: str
    session_secret: str
    session_hours: int
    max_upload_mb: int
    rate_limit_per_minute: int
    enable_pdf_ocr: bool
    pdf_ocr_languages: str
    test_mode: bool

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_password)


def load_settings() -> Settings:
    password = os.getenv("KB_PASSWORD", "change-me")
    session_secret = os.getenv("KB_SESSION_SECRET") or hashlib.sha256(password.encode("utf-8")).hexdigest()
    return Settings(
        auth_password=password,
        session_secret=session_secret,
        session_hours=int(os.getenv("KB_SESSION_HOURS", "12")),
        max_upload_mb=int(os.getenv("KB_MAX_UPLOAD_MB", "64")),
        rate_limit_per_minute=int(os.getenv("KB_RATE_LIMIT_PER_MINUTE", "120")),
        enable_pdf_ocr=_env_flag("KB_ENABLE_PDF_OCR", True),
        pdf_ocr_languages=os.getenv("KB_PDF_OCR_LANGS", "eng"),
        test_mode=_env_flag("KB_TEST_MODE", False),
    )
