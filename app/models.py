from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


BuildStatus = Literal["queued", "running", "completed", "failed"]


class DocumentVersion(BaseModel):
    version: int
    sha256: str
    stored_path: str
    text_path: str
    size_bytes: int
    uploaded_at: datetime
    extracted_chars: int = 0
    ocr_used: bool = False


class DocumentRecord(BaseModel):
    id: str
    name: str
    original_name: str
    extension: str
    file_type: str
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    active_version: int = 1
    latest_sha256: str = ""
    duplicate_uploads: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    versions: list[DocumentVersion] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    id: str
    document_id: str
    document_name: str
    document_source: str = ""
    document_tags: list[str] = Field(default_factory=list)
    version: int
    chunk_index: int
    text: str
    token_count: int
    char_count: int
    embedding: list[float] = Field(default_factory=list)


class BuildJobRecord(BaseModel):
    id: str
    status: BuildStatus = "queued"
    queued_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: float = 0.0
    message: str = "queued"
    auto: bool = False
    trigger: str = "manual"
    error: str | None = None
    gguf_filename: str | None = None
    gguf_path: str | None = None
    gguf_size_bytes: int = 0
    dirty_document_ids: list[str] = Field(default_factory=list)
    partial_rebuild: bool = False
    reused_chunks: int = 0
    rebuilt_chunks: int = 0
    removed_chunks: int = 0


class BuildInfo(BaseModel):
    in_progress: bool = False
    progress: float = 0.0
    message: str = "idle"
    last_built_at: datetime | None = None
    last_error: str | None = None
    current_filename: str | None = None
    current_path: str | None = None
    size_bytes: int = 0
    current_job_id: str | None = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    source: str
    tags: list[str] = Field(default_factory=list)
    version: int
    chunk_index: int
    token_count: int
    score: float
    text: str


class HealthInfo(BaseModel):
    ok: bool
    model_loaded: bool
    documents: int
    chunks: int
    current_job_id: str | None = None
    auth_enabled: bool = True
    partial_rebuild_supported: bool = True


class AdminState(BaseModel):
    password_hash: str = ""
    password_salt: str = ""
    password_changed_at: datetime | None = None


class KnowledgeBaseState(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    documents: list[DocumentRecord] = Field(default_factory=list)
    chunks: list[ChunkRecord] = Field(default_factory=list)
    tfidf_vocabulary: dict[str, int] = Field(default_factory=dict)
    tfidf_idf: dict[str, float] = Field(default_factory=dict)
    build: BuildInfo = Field(default_factory=BuildInfo)
    build_jobs: list[BuildJobRecord] = Field(default_factory=list)


def model_to_jsonable(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")
