"""
Strategy 3: Structure-aware chunking (paragraph / heading based)
--------------------------------------------------------------------
Instead of chunking by size at all, chunk by the document's own
logical structure: paragraphs, or markdown headings/sections.

The idea: a paragraph (or a section under a heading) is usually
already a self-contained "unit of meaning" that a human author
intended to be read together. So rather than imposing an artificial
character limit, we respect the author's structure — and only fall
back to size-based splitting if a single paragraph/section is too
large to embed well.

This tends to produce the most semantically coherent chunks, at the
cost of variable (sometimes unpredictable) chunk sizes — which
matters if you're paying per-token for embeddings or have strict
context window budgets downstream.

When to use: well-structured docs (technical docs, markdown READMEs,
docs with clear headings). Less useful on unstructured text (raw
transcripts, OCR'd scans) where structure isn't reliably present.
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_oversized(text: str, max_size: int) -> list[str]:
    """Fallback: if a paragraph/section is still too big, cut on
    sentence boundaries as a secondary pass."""
    if len(text) <= max_size:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, current = [], ""
    for s in sentences:
        candidate = (current + " " + s).strip() if current else s
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = s
    if current:
        pieces.append(current)
    return pieces


def chunk_structure_aware(
    text: str,
    max_chunk_size: int = 800,
) -> list[Chunk]:
    # Detect markdown-style headings; if present, section by heading.
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
    headings = list(heading_pattern.finditer(text))

    chunks: list[Chunk] = []
    idx = 0

    if headings:
        # Split by heading boundaries
        for i, match in enumerate(headings):
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            section_text = text[start:end].strip()
            heading_title = match.group(2)

            for piece in _split_oversized(section_text, max_chunk_size):
                chunks.append(Chunk(
                    text=piece,
                    metadata={
                        "strategy": "structure_aware",
                        "chunk_index": idx,
                        "section": heading_title,
                    },
                ))
                idx += 1
    else:
        # No headings: fall back to paragraph splitting
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            for piece in _split_oversized(para, max_chunk_size):
                chunks.append(Chunk(
                    text=piece,
                    metadata={"strategy": "structure_aware", "chunk_index": idx},
                ))
                idx += 1

    return chunks
