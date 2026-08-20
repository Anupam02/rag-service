"""
Step 4: Vector Store (ChromaDB)
---------------------------------
Goal: persist embedded chunks somewhere, and be able to ask
"give me the k chunks whose vectors are closest to this query vector."

Why not just a Python list + a loop computing cosine similarity
against everything? At small scale (hundreds of chunks) that's
literally fine and what ChromaDB does under the hood for small
collections anyway. The value of a real vector DB shows up at scale:

  - Approximate Nearest Neighbor (ANN) indexing (HNSW) so search is
    sub-linear instead of O(n) against every stored vector
  - Persistence to disk (survives restarts) instead of living only
    in RAM
  - Metadata filtering (e.g. "only search chunks from this source
    document") without re-embedding or re-scanning everything
  - A stable interface so you can swap ChromaDB for pgvector/Pinecone
    later without rewriting the retrieval layer

Design decision: we store metadata alongside each vector (source
filename, chunk_index) so retrieved results can be traced back to
"which document, which part" for citation in the final answer.

Distance metric: we explicitly set ChromaDB's collection to use
cosine distance. Since our embedder already L2-normalizes vectors,
this matches Euclidean/dot-product ranking too - but being explicit
here avoids a subtle bug: ChromaDB's DEFAULT metric is actually
squared L2 (not cosine), and while the ranking is equivalent for
normalized vectors, the raw distance NUMBERS you get back differ
significantly between metrics. If you ever show a "similarity score"
to a user or use a hard threshold to filter results, get this wrong
and your thresholds silently stop making sense.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid


@dataclass
class SearchResult:
    text: str
    metadata: dict[str, Any]
    score: float  # higher = more similar (we normalize this, see below)


class VectorStore:
    def __init__(
        self,
        collection_name: str = "rag_chunks",
        persist_dir: str = "./chroma_db",
    ):
        import chromadb

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # cosine distance explicitly - see docstring on why this matters
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_embedded_chunks(self, embedded_chunks: list) -> None:
        ids = [str(uuid.uuid4()) for _ in embedded_chunks]
        documents = [ec.text for ec in embedded_chunks]
        embeddings = [ec.vector.tolist() for ec in embedded_chunks]
        # Chroma metadata values must be str/int/float/bool, not None/dict
        metadatas = [
            {k: v for k, v in ec.metadata.items() if isinstance(v, (str, int, float, bool))}
            or {"_empty": True}
            for ec in embedded_chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query_vector, top_k: int = 5) -> list[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
        )

        out = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]  # cosine DISTANCE, not similarity

        for doc, meta, dist in zip(docs, metas, distances):
            # cosine distance = 1 - cosine similarity, so convert back
            # to similarity for an intuitive "higher = better" score
            similarity = 1 - dist
            out.append(SearchResult(text=doc, metadata=meta, score=similarity))

        return out

    def count(self) -> int:
        return self.collection.count()
