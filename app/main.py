from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.routes import router
from app.services.kb import kb_service


app = FastAPI(title="GGUF Knowledge Base Builder")
app.include_router(router)


@app.on_event("startup")
async def startup_event() -> None:
    await kb_service.startup()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return kb_service.render_index_html()
