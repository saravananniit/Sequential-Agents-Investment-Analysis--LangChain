"""FastAPI server: serves index.html and streams the pipeline progress.

Run:  uvicorn app:app --reload --port 8000
"""
import json
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents import check_config, run_pipeline

BASE = Path(__file__).parent
app = FastAPI(title="Sequential Agents - Investment Analysis")


class Config(BaseModel):
    provider: Literal["azure", "openai"] = "azure"
    api_key: str
    # Azure OpenAI
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = "2025-02-01-preview"
    # Plain OpenAI (or any OpenAI-compatible endpoint)
    model: Optional[str] = "gpt-4o"
    base_url: Optional[str] = None
    # Shared
    tavily_api_key: str
    temperature: float = Field(0.7, ge=0, le=2)
    max_iterations: int = Field(15, ge=10, le=60)


class RunRequest(BaseModel):
    config: Config
    question: str


@app.post("/api/check")
def check(cfg: Config):
    return check_config(cfg.model_dump())


@app.post("/api/run")
def run(req: RunRequest):
    def stream():
        for event in run_pipeline(req.config.model_dump(), req.question):
            yield json.dumps(event) + "\n"   # NDJSON, one event per line

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
