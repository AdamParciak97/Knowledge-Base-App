from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import mimetypes
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import secrets

import numpy as np
from fastapi import HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer

from app.config import Settings, load_settings
from app.models import AdminState, BuildInfo, BuildJobRecord, DocumentRecord, DocumentVersion, HealthInfo, KnowledgeBaseState, model_to_jsonable


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".env",
    ".csv",
    ".xml",
    ".html",
    ".py",
    ".js",
    ".ts",
    ".sql",
    ".log",
    ".cfg",
    ".conf",
    ".zip",
}
SAFE_GENERIC_MIME_TYPES = {"", "application/octet-stream"}
SAFE_MIME_TYPES = {
    "application/pdf",
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/toml",
    "application/sql",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/zip",
    "application/x-zip-compressed",
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/xml",
    "text/html",
    "text/x-python",
    "text/javascript",
    "application/javascript",
    "text/x-typescript",
    "text/x-log",
}
TOKEN_PATTERN = re.compile(r"\S+")
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIR_NAME = MODEL_NAME
SESSION_COOKIE = "kb_session"


@dataclass(slots=True)
class ExtractedContent:
    text: str
    ocr_used: bool = False


class FakeEmbeddingModel:
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(64, dtype=np.float32)
            for token in TOKEN_PATTERN.findall(text.lower()):
                idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % len(vector)
                vector[idx] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / norm
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


class KnowledgeBaseBase:
    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.base_dir / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.texts_dir = self.data_dir / "texts"
        self.gguf_dir = self.data_dir / "gguf"
        self.state_path = self.data_dir / "kb_state.json"
        self.admin_path = self.data_dir / "admin_state.json"
        self.template_path = self.base_dir / "app" / "templates" / "index.html"
        self.model_dir = self.base_dir / "models" / MODEL_DIR_NAME
        self.settings: Settings = load_settings()
        self.state = KnowledgeBaseState()
        self.admin_state = AdminState()
        self.model: SentenceTransformer | FakeEmbeddingModel | None = None
        self.tokenizer: Any = None
        self._build_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._queue_lock = asyncio.Lock()
        self._job_queue: asyncio.Queue[str] = asyncio.Queue()
        self._build_loop_task: asyncio.Task | None = None
        self._startup_done = False
        self._rate_limit: dict[str, deque[float]] = {}

    def ensure_ready(self) -> None:
        if self._startup_done:
            return
        self._ensure_dirs()
        self._load_state()
        self._load_admin_state()
        self._load_embedding_model()
        self._startup_done = True

    async def startup(self) -> None:
        self.ensure_ready()
        self._ensure_build_loop()
        if self.state.documents and self.get_download_path() is None:
            await self.enqueue_build(auto=True, trigger="startup")

    def _ensure_dirs(self) -> None:
        for path in (self.data_dir, self.uploads_dir, self.texts_dir, self.gguf_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _load_embedding_model(self) -> None:
        if self.settings.test_mode:
            self.model = FakeEmbeddingModel()
            self.tokenizer = None
            return
        model_path = str(self.model_dir) if self.model_dir.exists() else MODEL_NAME
        self.model = SentenceTransformer(model_path)
        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None and hasattr(self.model, "_first_module"):
            first_module = self.model._first_module()
            self.tokenizer = getattr(first_module, "tokenizer", None)

    def _migrate_state_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        migrated = dict(payload)
        migrated.setdefault("documents", [])
        migrated.setdefault("chunks", [])
        migrated.setdefault("tfidf_vocabulary", {})
        migrated.setdefault("tfidf_idf", {})
        migrated.setdefault("build_jobs", [])
        build = migrated.setdefault("build", {})
        build.setdefault("current_job_id", None)
        for document in migrated["documents"]:
            if "versions" in document:
                continue
            version = {
                "version": 1,
                "sha256": document.get("latest_sha256", ""),
                "stored_path": document.get("stored_path", ""),
                "text_path": document.get("text_path", ""),
                "size_bytes": document.get("size_bytes", 0),
                "uploaded_at": document.get("uploaded_at", datetime.now(UTC).isoformat()),
                "extracted_chars": 0,
                "ocr_used": False,
            }
            document["file_type"] = document.get("file_type") or self._classify_file_type(document.get("extension", ""))
            document["source"] = document.get("source", "")
            document["tags"] = document.get("tags", [])
            document["created_at"] = document.get("created_at") or document.get("uploaded_at") or datetime.now(UTC).isoformat()
            document["active_version"] = document.get("active_version", 1)
            document["latest_sha256"] = document.get("latest_sha256", "")
            document["duplicate_uploads"] = document.get("duplicate_uploads", 0)
            document["chunk_count"] = document.get("chunk_count", len(document.get("chunk_ids", [])))
            document["versions"] = [version]
            document.pop("stored_path", None)
            document.pop("text_path", None)
            document.pop("size_bytes", None)
            document.pop("uploaded_at", None)
        for index, chunk in enumerate(migrated["chunks"]):
            chunk.setdefault("document_source", "")
            chunk.setdefault("document_tags", [])
            chunk.setdefault("version", 1)
            chunk.setdefault("chunk_index", index)
            chunk.setdefault("char_count", len(chunk.get("text", "")))
        return migrated

    def _load_state(self) -> None:
        if not self.state_path.exists():
            self._persist_state_sync()
            return
        with self.state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.state = KnowledgeBaseState.model_validate(self._migrate_state_payload(payload))
        self._refresh_build_size()

    def _persist_state_sync(self) -> None:
        self.state.updated_at = datetime.now(UTC)
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(model_to_jsonable(self.state), handle, ensure_ascii=False, indent=2)

    async def _persist_state(self) -> None:
        async with self._state_lock:
            self._persist_state_sync()

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 150000).hex()

    def _load_admin_state(self) -> None:
        if self.admin_path.exists():
            with self.admin_path.open("r", encoding="utf-8") as handle:
                self.admin_state = AdminState.model_validate(json.load(handle))
            return
        salt = secrets.token_hex(16)
        self.admin_state = AdminState(
            password_salt=salt,
            password_hash=self._hash_password(self.settings.auth_password, salt),
            password_changed_at=None,
        )
        self._persist_admin_state_sync()

    def _persist_admin_state_sync(self) -> None:
        with self.admin_path.open("w", encoding="utf-8") as handle:
            json.dump(model_to_jsonable(self.admin_state), handle, ensure_ascii=False, indent=2)

    async def _persist_admin_state(self) -> None:
        async with self._state_lock:
            self._persist_admin_state_sync()

    def render_index_html(self) -> HTMLResponse:
        return HTMLResponse(self.template_path.read_text(encoding="utf-8"))

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _guess_content_type(self, filename: str, content_type: str, suffix: str) -> str:
        normalized = (content_type or "").lower().strip()
        if normalized:
            return normalized
        guessed, _ = mimetypes.guess_type(filename or f"file{suffix}")
        return (guessed or "").lower()

    def _normalize_tags(self, raw_tags: str | None) -> list[str]:
        if not raw_tags:
            return []
        tags: list[str] = []
        for part in raw_tags.split(","):
            tag = part.strip().lower()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _classify_file_type(self, suffix: str) -> str:
        mapping = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".ini": "config",
            ".env": "config",
            ".cfg": "config",
            ".conf": "config",
            ".csv": "csv",
            ".xml": "xml",
            ".html": "html",
            ".py": "code",
            ".js": "code",
            ".ts": "code",
            ".sql": "sql",
            ".log": "log",
            ".md": "markdown",
            ".txt": "text",
        }
        return mapping.get(suffix, "text")

    def _serialize_document(self, document: DocumentRecord) -> dict[str, Any]:
        payload = model_to_jsonable(document)
        active = self._get_active_version(document)
        payload["active_version_info"] = model_to_jsonable(active) if active else None
        return payload

    def _get_active_version(self, document: DocumentRecord) -> DocumentVersion | None:
        return next((version for version in document.versions if version.version == document.active_version), None)

    def _refresh_build_size(self) -> None:
        if not self.state.build.current_path:
            self.state.build.size_bytes = 0
            return
        target = self.base_dir / self.state.build.current_path
        self.state.build.size_bytes = target.stat().st_size if target.exists() else 0

    def get_status(self) -> dict:
        self.ensure_ready()
        build = BuildInfo.model_validate(model_to_jsonable(self.state.build))
        jobs = [model_to_jsonable(job) for job in sorted(self.state.build_jobs, key=lambda item: item.queued_at, reverse=True)]
        return {
            "documents": len(self.state.documents),
            "chunks": len(self.state.chunks),
            "gguf_size_bytes": build.size_bytes,
            "gguf_filename": build.current_filename,
            "build": model_to_jsonable(build),
            "build_jobs": jobs,
            "queue_depth": self._job_queue.qsize(),
        }

    def list_documents(self) -> list[dict]:
        self.ensure_ready()
        return [self._serialize_document(document) for document in self.state.documents]

    def list_build_jobs(self) -> list[dict]:
        self.ensure_ready()
        return [model_to_jsonable(job) for job in sorted(self.state.build_jobs, key=lambda item: item.queued_at, reverse=True)]

    def get_download_path(self) -> Path | None:
        self.ensure_ready()
        if not self.state.build.current_path:
            return None
        target = self.base_dir / self.state.build.current_path
        return target if target.exists() else None

    def verify_password(self, password: str) -> bool:
        self.ensure_ready()
        hashed = self._hash_password(password, self.admin_state.password_salt)
        return hmac.compare_digest(self.admin_state.password_hash, hashed)

    def _build_session_token(self, expires_at: int) -> str:
        payload = f"{expires_at}".encode("utf-8")
        signature = hmac.new(self.settings.session_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        raw = f"{expires_at}:{signature}".encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii")

    def _parse_session_token(self, token: str | None) -> bool:
        if not token:
            return False
        try:
            decoded = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
            expires_at_str, signature = decoded.split(":", 1)
            expires_at = int(expires_at_str)
        except Exception:
            return False
        if expires_at < int(time.time()):
            return False
        expected = hmac.new(self.settings.session_secret.encode("utf-8"), expires_at_str.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)

    def login(self, response: Response, password: str) -> dict:
        self.ensure_ready()
        if not self.verify_password(password):
            raise HTTPException(status_code=401, detail="Invalid password")
        expires_at = int(time.time()) + (self.settings.session_hours * 3600)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=self._build_session_token(expires_at),
            httponly=True,
            samesite="lax",
            max_age=self.settings.session_hours * 3600,
        )
        return {"authenticated": True, "expires_at": expires_at}

    def logout(self, response: Response) -> dict:
        response.delete_cookie(SESSION_COOKIE)
        return {"authenticated": False}

    def is_authenticated(self, request: Request) -> bool:
        self.ensure_ready()
        if not self.settings.auth_enabled:
            return True
        return self._parse_session_token(request.cookies.get(SESSION_COOKIE))

    def get_session_info(self, request: Request) -> dict:
        self.ensure_ready()
        return {
            "authenticated": self.is_authenticated(request),
            "auth_enabled": self.settings.auth_enabled,
            "max_upload_mb": self.settings.max_upload_mb,
            "rate_limit_per_minute": self.settings.rate_limit_per_minute,
            "default_password_warning": self.verify_password("change-me"),
        }

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    def enforce_rate_limit(self, request: Request) -> None:
        limit = self.settings.rate_limit_per_minute
        if limit <= 0:
            return
        now = time.time()
        bucket = self._rate_limit.setdefault(self._client_key(request), deque())
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        bucket.append(now)

    def require_access(self, request: Request) -> None:
        self.enforce_rate_limit(request)
        if not self.is_authenticated(request):
            raise HTTPException(status_code=401, detail="Authentication required")

    def health(self) -> dict:
        info = HealthInfo(
            ok=True,
            model_loaded=self.model is not None,
            documents=len(self.state.documents),
            chunks=len(self.state.chunks),
            current_job_id=self.state.build.current_job_id,
            auth_enabled=self.settings.auth_enabled,
            partial_rebuild_supported=True,
        )
        return model_to_jsonable(info)

    async def change_password(self, current_password: str, new_password: str) -> dict:
        self.ensure_ready()
        if len(new_password) < 8:
            raise HTTPException(status_code=400, detail="New password must have at least 8 characters")
        if not self.verify_password(current_password):
            raise HTTPException(status_code=401, detail="Current password is invalid")
        salt = secrets.token_hex(16)
        self.admin_state.password_salt = salt
        self.admin_state.password_hash = self._hash_password(new_password, salt)
        self.admin_state.password_changed_at = datetime.now(UTC)
        self.settings.session_secret = hashlib.sha256(f"{new_password}:{salt}".encode("utf-8")).hexdigest()
        await self._persist_admin_state()
        return {"changed": True, "password_changed_at": self.admin_state.password_changed_at.isoformat()}

    async def clear_build_history(self) -> dict:
        self.ensure_ready()
        current_path = self.state.build.current_path
        for gguf_file in self.gguf_dir.glob("knowledge_base_*.gguf"):
            if current_path and gguf_file == self.base_dir / current_path:
                continue
            gguf_file.unlink(missing_ok=True)
        kept = None
        if self.state.build_jobs:
            kept = self.state.build_jobs[-1]
        self.state.build_jobs = [kept] if kept else []
        await self._persist_state()
        return {"cleared": True, "kept_current_artifact": bool(current_path)}

    async def reset_rate_limits(self) -> dict:
        self._rate_limit.clear()
        return {"cleared": True}

    def diagnostics(self) -> dict:
        current_gguf = self.get_download_path()
        return {
            "model_loaded": self.model is not None,
            "model_dir_exists": self.model_dir.exists(),
            "tokenizer_loaded": self.tokenizer is not None,
            "ocr_enabled": self.settings.enable_pdf_ocr,
            "ocr_languages": self.settings.pdf_ocr_languages,
            "documents": len(self.state.documents),
            "chunks": len(self.state.chunks),
            "build_jobs": len(self.state.build_jobs),
            "queue_depth": self._job_queue.qsize(),
            "rate_limit_entries": len(self._rate_limit),
            "current_gguf": str(current_gguf) if current_gguf else None,
            "current_gguf_size": self.state.build.size_bytes,
            "password_changed_at": self.admin_state.password_changed_at.isoformat() if self.admin_state.password_changed_at else None,
        }
