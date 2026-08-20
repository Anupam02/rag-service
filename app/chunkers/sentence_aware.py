"""
Strategy 2: Sentence-aware (recursive) chunking
------------------------------------------------
Instead of cutting at a raw character count, we try splitting on
"nice" boundaries first — paragraphs, then sentences, then words —
falling back to a harder split only if a piece is still too big.

This is the "RecursiveCharacterTextSplitter" idea from LangChain,
implemented from scratch so you can see exactly what it does:

1. Try splitting the text on "\n\n" (paragraph breaks)
2. Any resulting piece that's still > chunk_size: split THAT piece
   on ". " (sentence breaks)
3. Any resulting piece still too big: split on " " (word breaks)
4. Any resulting piece still too big: hard character cut (last resort)

Then we greedily pack these pieces back together up to chunk_size,
so you don't end up with a "chunk" that's just one short sentence.

Why this matters: a chunk boundary that respects sentence structure
means each chunk is much more likely to be a semantically complete
thought — which is exactly what you want the embedding model to
encode into a vector.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_text(text: str, separators: list[str]) -> list[str]:
    """Recursively split text on the first separator that actually
    breaks it into multiple pieces, falling through to the next
    separator for any piece still too large."""
    if not separators:
        return [text]

    sep = separators[0]
    remaining_seps = separators[1:]

    if sep == "":
        # last resort: no separator left, return as-is
        return [text]

    pieces = text.split(sep)
    return [p for p in pieces if p.strip() != ""]


def chunk_sentence_aware(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_recursive(t: str, seps: list[str]) -> list[str]:
        if len(t) <= chunk_size or not seps:
            return [t]
        sep = seps[0]
        if sep == "":
            # hard cut, no natural boundary available
            return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)]

        pieces = [p for p in t.split(sep) if p.strip()]
        if len(pieces) == 1:
            # this separator didn't help, try the next one
            return split_recursive(t, seps[1:])

        result = []
        for p in pieces:
            if len(p) > chunk_size:
                result.extend(split_recursive(p, seps[1:]))
            else:
                result.append(p)
        return result

    atomic_pieces = split_recursive(text, separators)

    # Greedily pack atomic pieces into chunks up to chunk_size,
    # carrying `overlap` characters from the end of one chunk into
    # the start of the next.
    chunks: list[Chunk] = []
    current = ""
    idx = 0

    for piece in atomic_pieces:
        candidate = (current + " " + piece).strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(Chunk(
                    text=current,
                    metadata={"strategy": "sentence_aware", "chunk_index": idx},
                ))
                idx += 1
                # start next chunk with overlap tail of previous chunk
                tail = current[-overlap:] if overlap else ""
                current = (tail + " " + piece).strip()
            else:
                current = piece

    if current:
        chunks.append(Chunk(
            text=current,
            metadata={"strategy": "sentence_aware", "chunk_index": idx},
        ))

    return chunks
