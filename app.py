from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analysis_service import analyze_image_portfolio, analyze_text_portfolio

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

EXAMPLE_IMAGE_MAP = {
    "low": FRONTEND_DIR / "examples" / "example_low.png",
    "medium": FRONTEND_DIR / "examples" / "example_medium.png",
    "high": FRONTEND_DIR / "examples" / "example_high.png",
}

app = FastAPI(
    title="AI Portfolio Risk Copilot",
    description="Multimodale Portfolio-Risikoanalyse mit Text- und Bildinput.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class TextAnalysisRequest(BaseModel):
    text: str


@app.get("/")
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze/text")
def analyze_text(payload: TextAnalysisRequest) -> JSONResponse:
    try:
        result = analyze_text_portfolio(payload.text)
        return JSONResponse({"success": True, "data": result})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc



@app.post("/api/analyze/example-image")
def analyze_example_image(
    example: str = Query(..., description="low, medium oder high"),
    question: str = Query("Analysiere mein Portfolio-Risiko."),
) -> JSONResponse:
    example_key = str(example).strip().lower()
    image_path = EXAMPLE_IMAGE_MAP.get(example_key)

    if image_path is None or not image_path.exists():
        raise HTTPException(status_code=404, detail="Beispielbild nicht gefunden.")

    try:
        result = analyze_image_portfolio(
            image_path=image_path,
            optional_question=question,
        )
        return JSONResponse({"success": True, "data": result})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze/image")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form("Analysiere mein Portfolio-Risiko."),
) -> JSONResponse:
    suffix = Path(image.filename or "portfolio.png").suffix or ".png"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await image.read())
            tmp_path = Path(tmp.name)

        result = analyze_image_portfolio(
            image_path=tmp_path,
            optional_question=question,
        )
        return JSONResponse({"success": True, "data": result})

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    finally:
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
