from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from gguf import GGUFWriter
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models import BuildJobRecord, ChunkRecord, model_to_jsonable
from app.services.kb_base import MODEL_NAME, TOKEN_PATTERN


class BuildMixin:
    def _ensure_build_loop(self) -> None:
        if self._build_loop_task and not self._build_loop_task.done():
            return
        self._build_loop_task = asyncio.create_task(self._build_loop(), name="kb-build-loop")

    async def enqueue_build(self, auto: bool, trigger: str, dirty_document_ids: list[str] | None = None) -> dict:
        self.ensure_ready()
        if not self.settings.test_mode:
            self._ensure_build_loop()
        async with self._queue_lock:
            dirty_document_ids = list(dict.fromkeys(dirty_document_ids or []))
            job = BuildJobRecord(
                id=uuid.uuid4().hex,
                auto=auto,
                trigger=trigger,
                dirty_document_ids=dirty_document_ids,
                partial_rebuild=bool(dirty_document_ids),
            )
            self.state.build_jobs.append(job)
            self.state.build_jobs = self.state.build_jobs[-25:]
            await self._job_queue.put(job.id)
            self.state.build.current_job_id = job.id
            if not self.state.build.in_progress:
                self.state.build.message = "queued"
                self.state.build.progress = 0.0
            await self._persist_state()
        if self.settings.test_mode:
            await self._drain_build_queue_for_tests()
        return {"status": "queued", "job": model_to_jsonable(job)}

    async def _build_loop(self) -> None:
        while True:
            job_id = await self._job_queue.get()
            await self._run_build_job(job_id)
            self._job_queue.task_done()

    async def _drain_build_queue_for_tests(self) -> None:
        while not self._job_queue.empty():
            job_id = await self._job_queue.get()
            await self._run_build_job(job_id)
            self._job_queue.task_done()

    def _find_job(self, job_id: str) -> BuildJobRecord | None:
        return next((job for job in self.state.build_jobs if job.id == job_id), None)

    async def _run_build_job(self, job_id: str) -> None:
        job = self._find_job(job_id)
        if job is None:
            return
        async with self._build_lock:
            self.state.build.in_progress = True
            self.state.build.progress = 0.0
            self.state.build.message = f"running {job.trigger}"
            self.state.build.last_error = None
            self.state.build.current_job_id = job.id
            job.status = "running"
            job.started_at = datetime.now(UTC)
            job.progress = 0.0
            job.message = "starting"
            await self._persist_state()
            try:
                await asyncio.to_thread(self._rebuild_sync, job)
            except Exception as exc:  # pragma: no cover
                job.status = "failed"
                job.finished_at = datetime.now(UTC)
                job.error = str(exc)
                job.message = "failed"
                self.state.build.in_progress = False
                self.state.build.progress = 0.0
                self.state.build.message = "failed"
                self.state.build.last_error = str(exc)
                await self._persist_state()
                return
            job.status = "completed"
            job.finished_at = datetime.now(UTC)
            job.progress = 100.0
            job.message = "completed"
            job.gguf_filename = self.state.build.current_filename
            job.gguf_path = self.state.build.current_path
            job.gguf_size_bytes = self.state.build.size_bytes
            self.state.build.in_progress = False
            self.state.build.progress = 100.0
            self.state.build.message = "completed"
            self.state.build.last_built_at = datetime.now(UTC)
            self._refresh_build_size()
            await self._persist_state()

    def _update_progress(self, job: BuildJobRecord, progress: float, message: str) -> None:
        job.progress = progress
        job.message = message
        self.state.build.progress = progress
        self.state.build.message = message

    def _rebuild_sync(self, job: BuildJobRecord) -> None:
        documents = list(self.state.documents)
        dirty_ids = set(job.dirty_document_ids)
        current_doc_ids = {document.id for document in documents}
        reusable_chunks = [
            chunk for chunk in self.state.chunks
            if chunk.document_id in current_doc_ids and chunk.document_id not in dirty_ids
        ]
        chunks: list[ChunkRecord] = list(reusable_chunks)
        total_documents = max(len(documents), 1)
        rebuilt_chunks = 0
        self._update_progress(job, 5.0, "partial rebuild" if dirty_ids else "chunking documents")
        for index, document in enumerate(documents, start=1):
            active = self._get_active_version(document)
            if active is None:
                continue
            if dirty_ids and document.id not in dirty_ids:
                document.chunk_ids = [chunk.id for chunk in reusable_chunks if chunk.document_id == document.id]
                document.chunk_count = len(document.chunk_ids)
                self._update_progress(job, 5.0 + (30.0 * index / total_documents), f"reused {document.original_name}")
                continue
            text_path = self.base_dir / active.text_path
            text = text_path.read_text(encoding="utf-8", errors="ignore") if text_path.exists() else ""
            doc_chunks = self._chunk_text(text, document, active.version)
            document.chunk_ids = [chunk.id for chunk in doc_chunks]
            document.chunk_count = len(doc_chunks)
            document.updated_at = datetime.now(UTC)
            chunks.extend(doc_chunks)
            rebuilt_chunks += len(doc_chunks)
            self._update_progress(job, 5.0 + (30.0 * index / total_documents), f"chunked {document.original_name}")

        job.reused_chunks = len(reusable_chunks)
        job.rebuilt_chunks = rebuilt_chunks
        job.removed_chunks = max(0, len(self.state.chunks) - len(reusable_chunks))
        self.state.chunks = chunks
        if not chunks:
            self.state.tfidf_vocabulary = {}
            self.state.tfidf_idf = {}
            self._write_empty_gguf(job)
            return

        self._update_progress(job, 45.0, "creating embeddings")
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embed_texts(texts)
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.embedding = embedding.astype(np.float32).tolist()

        self._update_progress(job, 72.0, "building tfidf")
        vectorizer = TfidfVectorizer(max_features=8192)
        vectorizer.fit(texts)
        sorted_terms = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
        self.state.tfidf_vocabulary = {term: int(index) for term, index in vectorizer.vocabulary_.items()}
        self.state.tfidf_idf = {term: float(vectorizer.idf_[index]) for term, index in sorted_terms}

        self._update_progress(job, 90.0, "writing gguf")
        self._write_gguf(documents, chunks, job)

    def _chunk_text(self, text: str, document, version: int) -> list[ChunkRecord]:
        token_ids = self._encode_tokens(text)
        if not token_ids:
            return []
        chunk_size = 512
        overlap = 64
        step = chunk_size - overlap
        chunks: list[ChunkRecord] = []
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + chunk_size]
            if not window:
                continue
            chunk_text = self._decode_tokens(window)
            chunks.append(
                ChunkRecord(
                    id=uuid.uuid4().hex,
                    document_id=document.id,
                    document_name=document.original_name,
                    document_source=document.source,
                    document_tags=list(document.tags),
                    version=version,
                    chunk_index=len(chunks),
                    text=chunk_text,
                    token_count=len(window),
                    char_count=len(chunk_text),
                )
            )
            if start + chunk_size >= len(token_ids):
                break
        return chunks

    def _encode_tokens(self, text: str) -> list[int] | list[str]:
        if self.tokenizer is None:
            return TOKEN_PATTERN.findall(text)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode_tokens(self, tokens: list[int] | list[str]) -> str:
        if self.tokenizer is None:
            return " ".join(str(token) for token in tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def _write_empty_gguf(self, job: BuildJobRecord) -> None:
        filename = self._build_filename()
        output_path = self.gguf_dir / filename
        writer = GGUFWriter(str(output_path), "knowledge-base")
        writer.add_string("kb.name", "Personal Knowledge Base")
        writer.add_string("kb.created_at", datetime.now(UTC).isoformat())
        writer.add_string("kb.embedding_model", MODEL_NAME)
        writer.add_string("kb.files_json", json.dumps([doc.original_name for doc in self.state.documents], ensure_ascii=False))
        writer.add_string("kb.tfidf_vocabulary_json", "{}")
        writer.add_string("kb.tfidf_idf_json", "{}")
        writer.add_uint32("kb.documents", len(self.state.documents))
        writer.add_uint32("kb.chunks", 0)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
        self._set_current_build_file(output_path, job)

    def _write_gguf(self, documents, chunks: list[ChunkRecord], job: BuildJobRecord) -> None:
        filename = self._build_filename()
        output_path = self.gguf_dir / filename
        embeddings = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float32)
        writer = GGUFWriter(str(output_path), "knowledge-base")
        writer.add_string("kb.name", "Personal Knowledge Base")
        writer.add_string("kb.created_at", datetime.now(UTC).isoformat())
        writer.add_string("kb.embedding_model", MODEL_NAME)
        writer.add_uint32("kb.documents", len(documents))
        writer.add_uint32("kb.chunks", len(chunks))
        writer.add_string("kb.files_json", json.dumps([doc.original_name for doc in documents], ensure_ascii=False))
        writer.add_string("kb.document_manifest_json", json.dumps([model_to_jsonable(doc) for doc in documents], ensure_ascii=False))
        writer.add_string("kb.build_jobs_json", json.dumps([model_to_jsonable(build_job) for build_job in self.state.build_jobs], ensure_ascii=False))
        writer.add_string("kb.tfidf_vocabulary_json", json.dumps(self.state.tfidf_vocabulary, ensure_ascii=False))
        writer.add_string("kb.tfidf_idf_json", json.dumps(self.state.tfidf_idf, ensure_ascii=False))
        self._add_array_if_not_empty(writer, "kb.chunk_ids", [chunk.id for chunk in chunks])
        self._add_array_if_not_empty(writer, "kb.chunk_document_ids", [chunk.document_id for chunk in chunks])
        self._add_array_if_not_empty(writer, "kb.chunk_document_names", [chunk.document_name for chunk in chunks])
        self._add_array_if_not_empty(writer, "kb.chunk_sources", [chunk.document_source for chunk in chunks])
        self._add_array_if_not_empty(writer, "kb.chunk_versions", [chunk.version for chunk in chunks])
        self._add_array_if_not_empty(writer, "kb.chunk_indexes", [chunk.chunk_index for chunk in chunks])
        self._add_array_if_not_empty(writer, "kb.chunk_texts", [chunk.text for chunk in chunks])
        writer.add_tensor("kb.embeddings", embeddings)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        self._set_current_build_file(output_path, job)

    def _add_array_if_not_empty(self, writer: GGUFWriter, key: str, values: list[str] | list[int]) -> None:
        if values:
            writer.add_array(key, values)

    def _set_current_build_file(self, output_path: Path, job: BuildJobRecord) -> None:
        for stale_file in self.gguf_dir.glob("knowledge_base_*.gguf"):
            if stale_file != output_path and stale_file.exists():
                stale_file.unlink()
        self.state.build.current_filename = output_path.name
        self.state.build.current_path = str(output_path.relative_to(self.base_dir))
        self.state.build.size_bytes = output_path.stat().st_size if output_path.exists() else 0
        job.gguf_filename = output_path.name
        job.gguf_path = str(output_path.relative_to(self.base_dir))
        job.gguf_size_bytes = self.state.build.size_bytes

    def _build_filename(self) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        return f"knowledge_base_{stamp}.gguf"

    def search(
        self,
        query: str,
        document_id: str | None = None,
        source: str | None = None,
        file_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> dict:
        self.ensure_ready()
        if not self.state.chunks:
            return {"query": query, "results": []}
        requested_tags = [tag.lower() for tag in (tags or []) if tag]
        allowed_document_ids = {
            document.id
            for document in self.state.documents
            if (document_id is None or document.id == document_id)
            and (source is None or document.source == source)
            and (file_type is None or document.file_type == file_type)
            and (not requested_tags or all(tag in document.tags for tag in requested_tags))
        }
        filtered_chunks = [chunk for chunk in self.state.chunks if chunk.document_id in allowed_document_ids]
        if not filtered_chunks:
            return {"query": query, "results": []}
        query_vector = self._embed_texts([query])[0]
        embeddings = np.asarray([chunk.embedding for chunk in filtered_chunks], dtype=np.float32)
        scores = embeddings @ query_vector
        top_indices = np.argsort(scores)[::-1][: max(1, min(limit, 25))]
        results = []
        for rank, index in enumerate(top_indices, start=1):
            chunk = filtered_chunks[int(index)]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "source": chunk.document_source,
                    "tags": list(chunk.document_tags),
                    "version": chunk.version,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "score": float(scores[int(index)]),
                    "text": chunk.text,
                }
            )
        return {
            "query": query,
            "filters": {
                "document_id": document_id,
                "source": source,
                "file_type": file_type,
                "tags": requested_tags,
                "limit": limit,
            },
            "results": results,
        }
