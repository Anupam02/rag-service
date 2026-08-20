"""
Step 3: Embedding
-------------------
Goal: turn a chunk of text into a fixed-length vector such that
semantically similar chunks land close together in vector space.

We use sentence-transformers/all-MiniLM-L6-v2:
  - 384-dimensional output vectors
  - Fast + lightweight (fits your 8GB RAM constraint easily)
  - Fine-tuned specifically so cosine similarity between outputs
    reflects semantic similarity (see our earlier conversation on
    why this differs from raw BERT embeddings)

Key implementation detail: we L2-normalize every vector (unit length)
at embedding time. This means:
  1. Cosine similarity and dot product become mathematically
     equivalent for these vectors -> we can use the cheaper dot
     product at search time without losing correctness.
  2. Euclidean distance between two normalized vectors is also a
     monotonic function of cosine similarity, so ranking is
     preserved even if a downstream tool defaults to L2.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class EmbeddedChunk:
    text: str
    vector: np.ndarray
    metadata: dict


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True -> unit-length vectors, see docstring above
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_chunks(self, chunks: list) -> list[EmbeddedChunk]:
        texts = [c.text for c in chunks]
        vectors = self.embed_texts(texts)
        return [
            EmbeddedChunk(text=c.text, vector=v, metadata=c.metadata)
            for c, v in zip(chunks, vectors)
        ]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Since vectors are already unit-normalized, this is just the dot product.
    # Written explicitly (not assuming normalization) so this function is
    # correct even if you reuse it elsewhere with non-normalized vectors.
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    embedder = Embedder()
    print(f"Model loaded. Embedding dimension: {embedder.dim}")

    samples = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",         # similar meaning to above
        "Quarterly revenue grew by 12 percent.",  # unrelated
    ]
    vecs = embedder.embed_texts(samples)

    print("\nPairwise cosine similarities:")
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sim = cosine_similarity(vecs[i], vecs[j])
            print(f"  [{i}] vs [{j}]: {sim:.4f}")
            print(f"      '{samples[i]}'  <->  '{samples[j]}'")
