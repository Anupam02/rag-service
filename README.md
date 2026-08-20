# RAG as a Service

A from-scratch Retrieval-Augmented Generation (RAG) system, built step by step
to understand every component rather than wrapping a framework like LangChain.
Runs entirely on local/open-source tooling — no external API keys required.

## Architecture

Two pipelines, sharing one embedding model:

```
INDEXING (POST /index)
  Document file -> Loader -> Chunker -> Embedder -> Vector Store (ChromaDB)

QUERYING (POST /query)
  Question -> Embedder -> Vector Store search -> Generator (Ollama) -> Answer
```

```
app/
├── loaders/          Step 1 — file -> plain text (.txt, .md, .pdf, .docx)
├── chunkers/          Step 2 — text -> chunks (3 strategies, see below)
├── embeddings/        Step 3 — chunks -> vectors (sentence-transformers)
├── store/             Step 4 — persist + search vectors (ChromaDB)
├── generation/         Step 5 — retrieved chunks + question -> answer (Ollama)
├── rag_pipeline.py    Glue: RAGPipeline class combining all of the above
└── main.py            Step 6 — FastAPI service exposing /index and /query
```

### Chunking strategies (`app/chunkers/`)

Three implementations exist so the tradeoffs are visible, not just described:

| Strategy | File | Respects meaning | Predictable size | Best for |
|---|---|---|---|---|
| Fixed-size | `fixed_size.py` | No — cuts anywhere | Yes | Unstructured text (logs) |
| Sentence-aware | `sentence_aware.py` | Mostly | Roughly | General-purpose fallback |
| Structure-aware | `structure_aware.py` | Best | No — varies | Structured docs (headings) |

`RAGPipeline` currently defaults to **structure-aware**. Run
`python3 app/chunkers/compare.py <file>` to see all three side by side on any
document, including flags for chunks that cut mid-sentence.

### Why cosine similarity + normalized embeddings

Embeddings are L2-normalized at creation time (`embedder.py`), which makes
cosine similarity, dot product, and Euclidean distance all produce identical
*rankings* (though different raw numbers). The vector store is explicitly
configured for cosine distance so retrieval scores stay interpretable.

## Setup

```bash
pip install -r requirements.txt

# Ollama must be running locally with a model pulled:
ollama serve
ollama pull llama3.2
```

## Running

**As a script** (useful for debugging the pipeline in isolation):
```bash
python3 app/rag_pipeline.py
```

**As a service:**
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive API docs: http://localhost:8000/docs

```bash
# Index a document
curl -X POST http://localhost:8000/index -F "file=@data/sample_docs/rag_notes.txt"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunk size and why does it matter?"}'
```

**Resetting the index** (currently required between test runs — see
Idempotency below):
```bash
rm -rf chroma_db
```

## Current limitations & future improvements

Roughly in the order I'd tackle them:

### Correctness / robustness (do these first)
- ~~**Idempotent indexing.**~~ **Done.** Chunk IDs are now derived
  deterministically from `(source, chunk_index)` via `make_chunk_id()` in
  `vector_store.py`, and indexing uses `upsert` + `reindex_source`
  (delete-then-upsert) instead of `add`. Re-indexing the same document no
  longer creates duplicates, and re-indexing a document that now produces
  *fewer* chunks correctly prunes the old extras instead of leaving them
  orphaned.
- **Minimum chunk size filter.** Very short "orphan" chunks (e.g. an
  isolated heading like `"RAG Architecture Notes"`) produce weak, noisy
  embeddings that occasionally win a retrieval slot by accident. Merge any
  chunk under ~50 characters into a neighbor before embedding.
- **Error handling around Ollama.** `Generator.generate()` currently lets a
  connection error (Ollama not running) propagate as a raw 500. Should catch
  this and return a clear "generation service unavailable" error instead.
- **Corrupt/empty file handling.** A zero-byte upload or a PDF that's
  actually a scanned image (no extractable text) currently produces a
  `Document` with empty text, which then silently produces zero chunks.
  Should validate and return a clear error instead of indexing nothing.
- **Config management.** Model names, chunk size, `top_k`, Ollama URL, and
  ChromaDB path are all hardcoded across files. Move to a single `Settings`
  object (pydantic `BaseSettings`) reading from environment variables /
  `.env`, so deployment doesn't mean editing source.

### Retrieval quality
- **Reranking.** Vector similarity alone is a coarse filter. Adding a
  cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) as a
  second pass over the top ~20 candidates before picking the final top-k
  meaningfully improves precision — cross-encoders see the query and chunk
  together, not as separately-embedded vectors.
- **Hybrid search.** Pure semantic search misses exact keyword/acronym
  matches (e.g. searching "CQRS" might not surface a chunk that only
  mentions it once). Combining vector search with BM25 keyword search
  (ChromaDB doesn't do this natively — would need a second index, e.g.
  `rank_bm25`) and merging results tends to outperform either alone.
- **Query rewriting.** Short/ambiguous user questions embed poorly. An LLM
  pre-pass that expands or rephrases the query before embedding (HyDE —
  Hypothetical Document Embeddings — is one established technique) often
  improves retrieval, at the cost of an extra LLM call per query.
- **Metadata filtering.** The vector store already carries `source` in
  metadata but nothing currently uses it — add the ability to scope a query
  to specific documents (e.g. "only search within contract.pdf").
- **Chunk size experimentation.** Currently fixed at 500 chars for indexing.
  Worth building an eval set (see below) to actually measure retrieval
  quality across different chunk sizes rather than guessing.

### Evaluation
- **No way to measure quality yet.** There's currently no test set of
  (question, expected-answer or expected-source-chunk) pairs to check
  whether a change (chunking strategy, embedding model, reranking) actually
  helps or hurts. Worth building a small labeled eval set from your own
  documents and scripting retrieval-precision / answer-quality checks
  before tuning further — otherwise every change is a guess.

### Service hardening
- **Streaming responses.** `/query` currently waits for the full Ollama
  response before returning. For longer answers, streaming tokens back
  (Ollama supports `"stream": true`) gives a much better perceived latency,
  especially over HTTP/SSE.
- **Authentication.** No auth on any endpoint currently — fine for local
  use, not for anything exposed beyond localhost.
- **Rate limiting / concurrency limits.** Ollama can only usefully serve
  one or two concurrent generations on a laptop; unbounded concurrent
  requests to `/query` will queue up or degrade badly. Add a semaphore or
  request queue.
- **Async I/O.** `pipeline.query()` and `pipeline.index_document()` run
  synchronously inside FastAPI's request handlers — fine at low traffic,
  but blocks the event loop under load. Worth moving to `async def` +
  running the blocking model calls in a thread pool executor.
- **Persistent job tracking for indexing.** Large document uploads block
  the HTTP request until fully indexed. A background task queue (even
  something simple like FastAPI's `BackgroundTasks`, or Celery for real
  scale) with a job-status endpoint would handle this better.

### Deployment
- **Dockerize.** Package the FastAPI app + dependencies into a container;
  run Ollama as a sidecar container or point at a remote Ollama instance.
- **Swap-in production vector store.** ChromaDB is great for prototyping;
  for larger scale or multi-instance deployment, pgvector (Postgres) or a
  managed vector DB (Pinecone, Qdrant Cloud) would handle concurrent writes
  and horizontal scaling better.
- **Observability.** No logging/tracing currently beyond print statements.
  Structured logging (what was retrieved, what was generated, latency per
  stage) is essential once this is a real service — especially for
  debugging *why* a particular answer was wrong (bad retrieval? bad
  generation? bad chunk?).

### Nice-to-haves
- **Multi-document / multi-turn conversation support** — currently each
  `/query` call is stateless with no chat history.
- **Source highlighting** — show *which part* of a retrieved chunk the
  answer actually drew from, not just the whole chunk.
- **Swappable embedding/generation backends** — the code already isolates
  these behind `Embedder`/`Generator` classes; formalizing this as an
  interface (so swapping `all-MiniLM-L6-v2` for OpenAI embeddings, or
  swapping Ollama for the Anthropic API, is a config change) would make
  the "as a service" part more literal.
