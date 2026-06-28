from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
import os

from api.routes.evaluate import router as evaluate_router
from api.routes.upload import router as upload_router

app = FastAPI(
    title="HireFlow API",
    description="Intelligent Candidate Search & Evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate_router, prefix="/api", tags=["evaluation"])
app.include_router(upload_router,   prefix="/api", tags=["upload"])


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """
    Serve test_ui.html from project root.
    Looks relative to current working directory (where uvicorn is run from).
    """
    # cwd = hireFlow/ (where you run uvicorn from)
    html_path = Path(os.getcwd()) / "test_ui.html"
    if not html_path.exists():
        return HTMLResponse(
            f"<h2>test_ui.html not found</h2><p>Looked in: {html_path}</p>",
            status_code=404
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "HireFlow API"}