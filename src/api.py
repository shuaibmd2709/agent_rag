import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi.responses import FileResponse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline_v6 import retrieve
from src.pipeline_v7 import ask

log = logging.getLogger("api")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class SourceChunk(BaseModel):
    doc: str
    article: str
    page: int
    text: str


class Answer(BaseModel):
    answer: str
    sources: str
    warning: str | None = None
    regenerated: bool
    contexts: list[SourceChunk]


@asynccontextmanager
async def lifespan(app: FastAPI):
    retrieve("warm up")  # loads the embedding model, reranker and index; no LLM call
    yield


app = FastAPI(title="Regulatory RAG: DORA and EU AI Act", lifespan=lifespan)
STATIC = Path(__file__).resolve().parent.parent / "static"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask_endpoint(body: Question):
    try:
        result = ask(body.question)
    except Exception:
        log.exception("ask failed")
        raise HTTPException(status_code=502, detail="The answer service failed. Please try again.")
    return Answer(
        answer=result["answer"],
        sources=result["sources"],
        warning=result["warning"],
        regenerated=result["regenerated"],
        contexts=[SourceChunk(**c) for c in result["contexts"]],
    )