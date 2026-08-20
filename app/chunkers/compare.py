"""
Compare all three chunking strategies on the same document.

Run: python3 app/chunkers/compare.py data/sample_docs/rag_notes.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.loaders.document_loader import load_document
from app.chunkers.fixed_size import chunk_fixed_size
from app.chunkers.sentence_aware import chunk_sentence_aware
from app.chunkers.structure_aware import chunk_structure_aware


def show(name: str, chunks: list) -> None:
    sizes = [len(c.text) for c in chunks]
    print(f"\n{'='*70}")
    print(f"{name}  |  {len(chunks)} chunks  |  sizes: {sizes}")
    print("=" * 70)
    for i, c in enumerate(chunks):
        preview = c.text.replace("\n", " ")[:120]
        cut_mid_sentence = c.text.strip() and c.text.strip()[-1] not in ".!?:\""
        flag = "  <-- cuts mid-sentence" if cut_mid_sentence else ""
        print(f"[{i}] ({len(c.text)} chars){flag}\n    {preview}...")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs/rag_notes.txt"
    doc = load_document(path)

    show("STRATEGY 1: Fixed-size (300 chars, 30 overlap)",
         chunk_fixed_size(doc.text, chunk_size=300, overlap=30))

    show("STRATEGY 2: Sentence-aware (300 chars, 30 overlap)",
         chunk_sentence_aware(doc.text, chunk_size=300, overlap=30))

    show("STRATEGY 3: Structure-aware (max 300 chars)",
         chunk_structure_aware(doc.text, max_chunk_size=300))
