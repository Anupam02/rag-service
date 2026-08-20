"""
End-to-end test: Load -> Chunk -> Embed -> Store -> Search

This is the first point where you see the whole retrieval half of
RAG working together. Run this after Step 4.

Run: python3 app/pipeline_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.loaders.document_loader import load_document
from app.chunkers.structure_aware import chunk_structure_aware
from app.embeddings.embedder import Embedder
from app.store.vector_store import VectorStore


def main():
    # 1. LOAD
    doc = load_document("data/sample_docs/rag_notes.txt")
    print(f"Loaded: {doc.source} ({len(doc.text)} chars)")

    # 2. CHUNK
    chunks = chunk_structure_aware(doc.text, max_chunk_size=300)
    print(f"Chunked into {len(chunks)} pieces")

    # 3. EMBED
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks (dim={embedder.dim})")

    # 4. STORE
    store = VectorStore(persist_dir="./chroma_db")
    store.add_embedded_chunks(embedded_chunks)
    print(f"Stored. Collection now has {store.count()} vectors total")

    # 4b. SEARCH - this is the retrieval step we'll wire into generation next
    query = "What determines how well retrieval works?"
    query_vector = embedder.embed_texts([query])[0]
    results = store.search(query_vector, top_k=3)

    print(f"\nQuery: '{query}'")
    print("Top matches:")
    for i, r in enumerate(results):
        print(f"\n[{i}] score={r.score:.4f}  section={r.metadata.get('section', '-')}")
        print(f"    {r.text[:150]}...")


if __name__ == "__main__":
    main()
