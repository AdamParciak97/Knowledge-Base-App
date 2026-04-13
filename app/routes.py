from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from app.services.kb import kb_service


router = APIRouter()

FAVICON_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
<rect width='64' height='64' rx='14' fill='#182223'/>
<path d='M16 18h24c5 0 9 4 9 9v19H25c-5 0-9-4-9-9V18z' fill='#af4f2c'/>
<path d='M24 24h14c3 0 6 3 6 6v10H30c-3 0-6-3-6-6V24z' fill='#fff4ec'/>
<circle cx='47' cy='47' r='9' fill='#2e6a4b'/>
<path d='M43 47h8M47 43v8' stroke='#fffdfa' stroke-width='2.5' stroke-linecap='round'/>
</svg>"""


def require_access(request: Request) -> None:
    kb_service.require_access(request)


@router.get("/health")
async def health() -> dict:
    return kb_service.health()


@router.get("/favicon.ico")
async def favicon() -> Response:
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@router.get("/auth/session")
async def auth_session(request: Request) -> dict:
    return kb_service.get_session_info(request)


@router.post("/auth/login")
async def auth_login(response: Response, password: str = Form(...)) -> dict:
    return kb_service.login(response, password)


@router.post("/auth/logout")
async def auth_logout(response: Response) -> dict:
    return kb_service.logout(response)


@router.get("/admin/diagnostics", dependencies=[Depends(require_access)])
async def admin_diagnostics() -> dict:
    return kb_service.diagnostics()


@router.post("/admin/password", dependencies=[Depends(require_access)])
async def admin_password(current_password: str = Form(...), new_password: str = Form(...)) -> dict:
    return await kb_service.change_password(current_password=current_password, new_password=new_password)


@router.post("/admin/clear-builds", dependencies=[Depends(require_access)])
async def admin_clear_builds() -> dict:
    return await kb_service.clear_build_history()


@router.post("/admin/reset-rate-limits", dependencies=[Depends(require_access)])
async def admin_reset_rate_limits() -> dict:
    return await kb_service.reset_rate_limits()


@router.post("/upload", dependencies=[Depends(require_access)])
async def upload(
    files: list[UploadFile] = File(...),
    source: str = Form(""),
    tags: str = Form(""),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    try:
        return await kb_service.upload_files(files, source=source, tags=tags)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/build", dependencies=[Depends(require_access)])
async def build() -> dict:
    return await kb_service.enqueue_build(auto=False, trigger="manual")


@router.get("/status", dependencies=[Depends(require_access)])
async def status() -> dict:
    return kb_service.get_status()


@router.get("/builds", dependencies=[Depends(require_access)])
async def builds() -> list[dict]:
    return kb_service.list_build_jobs()


@router.get("/documents", dependencies=[Depends(require_access)])
async def documents() -> list[dict]:
    return kb_service.list_documents()


@router.get("/document/{document_id}/chunks", dependencies=[Depends(require_access)])
async def document_chunks(document_id: str) -> list[dict]:
    try:
        return kb_service.get_document_chunks(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc


@router.delete("/document/{document_id}", dependencies=[Depends(require_access)])
async def delete_document(document_id: str) -> dict:
    try:
        return await kb_service.delete_document(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc


@router.get("/download", dependencies=[Depends(require_access)])
async def download() -> FileResponse:
    file_path = kb_service.get_download_path()
    if not file_path:
        raise HTTPException(status_code=404, detail="GGUF file not built yet")
    return FileResponse(path=file_path, filename=file_path.name, media_type="application/octet-stream")


@router.get("/search", dependencies=[Depends(require_access)])
async def search(
    q: str = Query(..., min_length=1),
    document_id: str | None = Query(default=None),
    source: str | None = Query(default=None),
    file_type: str | None = Query(default=None),
    tags: list[str] = Query(default=[]),
    limit: int = Query(default=10, ge=1, le=25),
) -> dict:
    return kb_service.search(
        query=q,
        document_id=document_id,
        source=source,
        file_type=file_type,
        tags=tags,
        limit=limit,
    )
