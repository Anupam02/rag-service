"""
Strategy 1: Fixed-size chunking (character count, with overlap)
------------------------------------------------------------------
The simplest possible approach: cut the text every N characters,
regardless of what's there — could slice a sentence, even a word,
right in half.

Why overlap? Without it, a sentence that straddles a chunk boundary
gets cut in two, and BOTH halves lose meaning:

    Chunk A: "...the algorithm's time complexity is O(n log n) when the"
    Chunk B: "input is already sorted, which is the best case scenario..."

Neither chunk alone captures "O(n log n) when input is already sorted."
Overlap (re-including the tail of chunk A at the start of chunk B)
reduces — but doesn't eliminate — this problem.

When to use: quick prototyping, or text with no real structure
(logs, transcripts). Rarely the best choice for production RAG.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_fixed_size(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)
    idx = 0

    while start < text_len:
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={"strategy": "fixed_size", "chunk_index": idx,
                          "char_start": start, "char_end": min(end, text_len)},
            )
        )
        idx += 1
        # advance start, but step back by `overlap` so consecutive
        # chunks share some text
        start += chunk_size - overlap

    return chunks
