from __future__ import annotations

import csv
import html
import io
import json
import re
import uuid
import zipfile
from configparser import ConfigParser
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import fitz
import toml
import yaml
from docx import Document as DocxDocument
from fastapi import UploadFile

from app.models import DocumentRecord, DocumentVersion, model_to_jsonable
from app.services.kb_base import ExtractedContent, SAFE_GENERIC_MIME_TYPES, SAFE_MIME_TYPES, SUPPORTED_EXTENSIONS


class DocumentMixin:
    def _validate_upload_metadata(self, filename: str, suffix: str, size_bytes: int, content_type: str) -> None:
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {filename}")
        if size_bytes > self.settings.max_upload_bytes:
            raise ValueError(f"File too large: {filename} exceeds {self.settings.max_upload_mb} MB")
        if content_type in SAFE_GENERIC_MIME_TYPES:
            return
        if content_type.startswith("text/"):
            return
        if content_type not in SAFE_MIME_TYPES:
            raise ValueError(f"Unsupported content type for {filename}: {content_type}")

    async def upload_files(self, files: list[UploadFile], source: str = "", tags: str = "") -> dict:
        self.ensure_ready()
        normalized_tags = self._normalize_tags(tags)
        uploaded_items: list[dict] = []
        changed = False
        dirty_document_ids: list[str] = []
        skipped: list[dict] = []
        for upload in files:
            file_results, state_changed, dirty_ids, skipped_items = await self._process_upload(upload, source=source.strip(), tags=normalized_tags)
            uploaded_items.extend(file_results)
            skipped.extend(skipped_items)
            for document_id in dirty_ids:
                if document_id not in dirty_document_ids:
                    dirty_document_ids.append(document_id)
            changed = changed or state_changed
        await self._persist_state()
        if changed:
            build = await self.enqueue_build(auto=True, trigger="upload", dirty_document_ids=dirty_document_ids)
        else:
            build = {"status": "skipped", "job": None}
        return {"uploaded": uploaded_items, "skipped": skipped, "build": build, "changed": changed}

    async def _process_upload(self, upload: UploadFile, source: str, tags: list[str]) -> tuple[list[dict], bool, list[str], list[dict]]:
        data = await upload.read()
        await upload.close()
        filename = upload.filename or "unknown"
        suffix = Path(filename).suffix.lower()
        content_type = self._guess_content_type(filename, upload.content_type or "", suffix)
        if suffix == ".zip":
            return self._process_zip_archive(filename, data, source, tags)
        result, changed, document_id = self._upsert_bytes(filename, data, content_type, source, tags)
        return [result], changed, ([document_id] if document_id else []), []

    def _process_zip_archive(self, archive_name: str, data: bytes, source: str, tags: list[str]) -> tuple[list[dict], bool, list[str], list[dict]]:
        uploaded: list[dict] = []
        skipped: list[dict] = []
        dirty_document_ids: list[str] = []
        changed = False
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                filename = Path(member.filename).name
                if not filename:
                    continue
                suffix = Path(filename).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS or suffix == ".zip":
                    skipped.append({"name": member.filename, "reason": "unsupported"})
                    continue
                file_data = archive.read(member)
                content_type = self._guess_content_type(filename, "", suffix)
                try:
                    result, state_changed, document_id = self._upsert_bytes(filename, file_data, content_type, source or member.filename, tags)
                except ValueError as exc:
                    skipped.append({"name": member.filename, "reason": str(exc)})
                    continue
                uploaded.append(result)
                changed = changed or state_changed
                if document_id and document_id not in dirty_document_ids:
                    dirty_document_ids.append(document_id)
        return uploaded, changed, dirty_document_ids, skipped

    def _upsert_bytes(self, filename: str, data: bytes, content_type: str, source: str, tags: list[str]) -> tuple[dict, bool, str | None]:
        suffix = Path(filename).suffix.lower()
        self._validate_upload_metadata(filename, suffix, len(data), content_type)
        sha256 = self._hash_bytes(data)
        existing_by_hash = next((doc for doc in self.state.documents if doc.latest_sha256 == sha256), None)
        if existing_by_hash is not None:
            existing_by_hash.duplicate_uploads += 1
            existing_by_hash.updated_at = datetime.now(UTC)
            if source:
                existing_by_hash.source = source
            if tags:
                existing_by_hash.tags = sorted(set(existing_by_hash.tags + tags))
            return {
                "action": "duplicate",
                "document_id": existing_by_hash.id,
                "document_name": existing_by_hash.original_name,
                "sha256": sha256,
            }, False, None

        target_document = next((doc for doc in self.state.documents if doc.original_name == filename), None)
        extracted = self._extract_text_bytes(data, suffix)
        now = datetime.now(UTC)

        if target_document is None:
            document_id = uuid.uuid4().hex
            record = DocumentRecord(
                id=document_id,
                name=Path(filename or document_id).stem,
                original_name=filename or f"{document_id}{suffix}",
                extension=suffix,
                file_type=self._classify_file_type(suffix),
                source=source,
                tags=list(tags),
                created_at=now,
                updated_at=now,
                active_version=1,
                latest_sha256=sha256,
            )
            version = self._write_document_version(record.id, 1, suffix, data, extracted, sha256, now)
            record.versions.append(version)
            self.state.documents.append(record)
            return {"action": "created", "document_id": record.id, "document_name": record.original_name, "sha256": sha256}, True, record.id

        version_number = max((version.version for version in target_document.versions), default=0) + 1
        version = self._write_document_version(target_document.id, version_number, suffix, data, extracted, sha256, now)
        target_document.extension = suffix
        target_document.file_type = self._classify_file_type(suffix)
        target_document.source = source or target_document.source
        target_document.tags = sorted(set(target_document.tags + tags))
        target_document.updated_at = now
        target_document.active_version = version_number
        target_document.latest_sha256 = sha256
        target_document.versions.append(version)
        return {
            "action": "updated",
            "document_id": target_document.id,
            "document_name": target_document.original_name,
            "version": version_number,
            "sha256": sha256,
        }, True, target_document.id

    def _write_document_version(
        self,
        document_id: str,
        version_number: int,
        suffix: str,
        data: bytes,
        extracted: ExtractedContent,
        sha256: str,
        uploaded_at: datetime,
    ) -> DocumentVersion:
        stored_path = self.uploads_dir / f"{document_id}_v{version_number}{suffix}"
        text_path = self.texts_dir / f"{document_id}_v{version_number}.txt"
        stored_path.write_bytes(data)
        text_path.write_text(extracted.text, encoding="utf-8")
        return DocumentVersion(
            version=version_number,
            sha256=sha256,
            stored_path=str(stored_path.relative_to(self.base_dir)),
            text_path=str(text_path.relative_to(self.base_dir)),
            size_bytes=len(data),
            uploaded_at=uploaded_at,
            extracted_chars=len(extracted.text),
            ocr_used=extracted.ocr_used,
        )

    def _extract_text_bytes(self, data: bytes, suffix: str) -> ExtractedContent:
        temp_path = self.data_dir / f".tmp_{uuid.uuid4().hex}{suffix}"
        temp_path.write_bytes(data)
        try:
            return self._extract_text(temp_path, suffix)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _extract_text(self, path: Path, suffix: str) -> ExtractedContent:
        readers = {
            ".pdf": self._read_pdf,
            ".docx": self._read_docx,
            ".yaml": self._read_structured_yaml,
            ".yml": self._read_structured_yaml,
            ".json": self._read_structured_json,
            ".toml": self._read_structured_toml,
            ".ini": self._read_structured_ini,
            ".env": self._read_text,
            ".csv": self._read_csv,
            ".xml": self._read_xml,
            ".html": self._read_html,
        }
        reader = readers.get(suffix, self._read_text)
        try:
            return reader(path)
        except Exception:
            return ExtractedContent(text=path.read_text(encoding="utf-8", errors="ignore").strip())

    def _read_text(self, path: Path) -> ExtractedContent:
        return ExtractedContent(text=path.read_text(encoding="utf-8", errors="ignore").strip())

    def _read_pdf(self, path: Path) -> ExtractedContent:
        texts: list[str] = []
        ocr_used = False
        with fitz.open(path) as document:
            for page in document:
                page_text = page.get_text("text").strip()
                if page_text:
                    texts.append(page_text)
                    continue
                if self.settings.enable_pdf_ocr:
                    ocr_text = self._try_pdf_ocr(page)
                    if ocr_text:
                        texts.append(ocr_text)
                        ocr_used = True
        return ExtractedContent(text="\n\n".join(texts).strip(), ocr_used=ocr_used)

    def _try_pdf_ocr(self, page: fitz.Page) -> str:
        try:
            if not hasattr(page, "get_textpage_ocr"):
                return ""
            text_page = page.get_textpage_ocr(language=self.settings.pdf_ocr_languages, full=True)
            return page.get_text("text", textpage=text_page).strip()
        except Exception:
            return ""

    def _read_docx(self, path: Path) -> ExtractedContent:
        doc = DocxDocument(str(path))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        return ExtractedContent(text=text)

    def _read_structured_yaml(self, path: Path) -> ExtractedContent:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
        return ExtractedContent(text=yaml.safe_dump(data, allow_unicode=False, sort_keys=False).strip())

    def _read_structured_json(self, path: Path) -> ExtractedContent:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return ExtractedContent(text=json.dumps(data, ensure_ascii=False, indent=2).strip())

    def _read_structured_toml(self, path: Path) -> ExtractedContent:
        data = toml.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return ExtractedContent(text=json.dumps(data, ensure_ascii=False, indent=2).strip())

    def _read_structured_ini(self, path: Path) -> ExtractedContent:
        parser = ConfigParser()
        parser.read(path, encoding="utf-8")
        data = {section: dict(parser.items(section)) for section in parser.sections()}
        return ExtractedContent(text=json.dumps(data, ensure_ascii=False, indent=2).strip())

    def _read_csv(self, path: Path) -> ExtractedContent:
        rows: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                rows.append(", ".join(row))
        return ExtractedContent(text="\n".join(rows).strip())

    def _read_xml(self, path: Path) -> ExtractedContent:
        root = ElementTree.fromstring(path.read_text(encoding="utf-8", errors="ignore"))
        return ExtractedContent(text=ElementTree.tostring(root, encoding="unicode", method="text").strip())

    def _read_html(self, path: Path) -> ExtractedContent:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        stripped = re.sub(r"<script.*?</script>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<style.*?</style>", " ", stripped, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        return ExtractedContent(text=html.unescape(stripped).strip())

    def get_document_chunks(self, document_id: str) -> list[dict]:
        self.ensure_ready()
        target = next((document for document in self.state.documents if document.id == document_id), None)
        if target is None:
            raise FileNotFoundError(document_id)
        chunks = [chunk for chunk in self.state.chunks if chunk.document_id == document_id]
        chunks.sort(key=lambda chunk: (chunk.version, chunk.chunk_index))
        return [model_to_jsonable(chunk) for chunk in chunks]

    async def delete_document(self, document_id: str) -> dict:
        self.ensure_ready()
        target = next((document for document in self.state.documents if document.id == document_id), None)
        if target is None:
            raise FileNotFoundError(document_id)
        for version in target.versions:
            for path_str in (version.stored_path, version.text_path):
                path = self.base_dir / path_str
                if path.exists():
                    path.unlink()
        self.state.documents = [document for document in self.state.documents if document.id != document_id]
        self.state.chunks = [chunk for chunk in self.state.chunks if chunk.document_id != document_id]
        await self._persist_state()
        build = await self.enqueue_build(auto=True, trigger="delete", dirty_document_ids=[document_id])
        return {"deleted": document_id, "build": build}
