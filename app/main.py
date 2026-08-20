"""
Step 6: FastAPI service
--------------------------
Wraps RAGPipeline in an HTTP API with two core endpoints:

  POST /index   - upload a document, index it into the vector store
  POST /query   - ask a question, get a grounded answer + sources

Key design decision: the RAGPipeline is created ONCE at app startup
(via FastAPI's lifespan handler), not per-request. Why this matters:
  - Embedder.__init__ loads a SentenceTransformer model into memory -
    this takes real time (a couple seconds) and RAM. Doing it per
    request would make every call slow and eventually OOM your
    8GB machine as multiple model copies pile up.
  - VectorStore holds a persistent connection to the on-disk ChromaDB.
    Reopening it per request is wasteful and can cause file-lock
    contention under concurrent requests.

This "load once, reuse across requests" pattern is the same reason
you don't reconnect to a database on every query in a normal backend
service - the RAG pipeline's expensive resources (model weights, DB
handle) are exactly analogous to a DB connection pool.

Run: uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs  (FastAPI auto-generates this)
"""

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.rag_pipeline import RAGPipeline

# Global pipeline instance - populated at startup, reused by every request
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading RAG pipeline (embedder, vector store, generator)...")
    pipeline = RAGPipeline()
    print("RAG pipeline ready.")
    yield
    # (nothing to clean up on shutdown for now - ChromaDB persists to disk)


app = FastAPI(title="RAG as a Service", lifespan=lifespan)


# ---------- Request/response schemas ----------
# Pydantic models here do two jobs: validate incoming request bodies,
# and auto-generate the OpenAPI docs at /docs.

class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceInfo(BaseModel):
    text: str
    source: str | None
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]


class IndexResponse(BaseModel):
    source: str
    num_chunks: int
    total_vectors_in_store: int


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.post("/index", response_model=IndexResponse)
async def index_document(file: UploadFile = File(...)):
    """Upload a document (.txt, .md, .pdf, .docx) and index it."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Save the uploaded file to a temp path - our loader works off
    # file paths, so we need it on disk momentarily, not just in memory.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = pipeline.index_document(tmp_path, source_name=file.filename)
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if request.top_k:
        pipeline.top_k = request.top_k
    result = pipeline.query(request.question)
    return result
