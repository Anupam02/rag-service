"""
The full RAG pipeline, end to end.

This class is deliberately just glue - each real piece of logic lives
in its own module (loaders/, chunkers/, embeddings/, store/, generation/).
That separation is what will let us swap pieces later (different
chunking strategy, different vector DB, different LLM) without
rewriting the pipeline itself - and it's what will let us expose this
as clean API endpoints in Step 6.

Two distinct phases, matching the two pipelines from day one:

  index_document(path)  -> the INDEXING pipeline (load, chunk, embed, store)
  query(question)        -> the QUERYING pipeline (embed, search, generate)
"""

import sys
from pathlib import Path

# rag_pipeline.py lives at app/rag_pipeline.py, so its parent is app/,
# and parent.parent is the project root - that's what needs to be on
# sys.path for "from app.loaders..." to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.loaders.document_loader import load_document
from app.chunkers.structure_aware import chunk_structure_aware
from app.embeddings.embedder import Embedder
from app.store.vector_store import VectorStore
from app.generation.generator import Generator


class RAGPipeline:
    def __init__(
        self,
        collection_name: str = "rag_chunks",
        persist_dir: str = "./chroma_db",
        chunk_size: int = 500,
        llm_model: str = "llama3.2",
        top_k: int = 3,
    ):
        self.embedder = Embedder()
        self.store = VectorStore(collection_name=collection_name, persist_dir=persist_dir)
        self.generator = Generator(model=llm_model)
        self.chunk_size = chunk_size
        self.top_k = top_k

    def index_document(self, path: str, source_name: str | None = None) -> dict:
        """INDEXING pipeline: load -> chunk -> embed -> store

        source_name: optional override for the citation label stored
        with each chunk. Needed when `path` is a temp file (e.g. an
        uploaded file saved to a random tmp path by the API layer) -
        without this, citations would show the meaningless temp
        filename instead of the document's real name.
        """
        doc = load_document(path)
        chunks = chunk_structure_aware(doc.text, max_chunk_size=self.chunk_size)

        # carry the source filename onto every chunk's metadata -
        # this is what lets the generator cite sources later
        label = source_name or doc.source
        for c in chunks:
            c.metadata["source"] = label

        embedded_chunks = self.embedder.embed_chunks(chunks)
        # delete-then-upsert: handles the case where re-indexing a
        # changed document now produces fewer chunks than before (see
        # the tradeoff note in vector_store.py's docstring)
        self.store.reindex_source(label, embedded_chunks)

        return {
            "source": label,
            "num_chunks": len(chunks),
            "total_vectors_in_store": self.store.count(),
        }

    def query(self, question: str) -> dict:
        """QUERYING pipeline: embed query -> search -> generate answer"""
        query_vector = self.embedder.embed_texts([question])[0]
        retrieved = self.store.search(query_vector, top_k=self.top_k)

        if not retrieved:
            return {
                "question": question,
                "answer": "No documents have been indexed yet.",
                "sources": [],
            }

        answer = self.generator.generate(question, retrieved)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"text": r.text[:150], "source": r.metadata.get("source"), "score": r.score}
                for r in retrieved
            ],
        }


if __name__ == "__main__":
    rag = RAGPipeline()

    result = rag.index_document("data/sample_docs/rag_notes.txt")
    print("Indexed:", result)

    answer = rag.query("What is chunk size and why does it matter?")
    print("\nQuestion:", answer["question"])
    print("Answer:", answer["answer"])
    print("\nSources used:")
    for s in answer["sources"]:
        print(f"  - {s['source']} (score={s['score']:.3f}): {s['text']}...")
