"""
Structure-Aware Text Chunking — implemented from scratch.

No LangChain splitters are used. This module implements a three-tier
chunking strategy that prioritises preserving semantic coherence over
arbitrary character-boundary slicing.

Why three tiers?
----------------
Naively splitting text every N characters breaks sentences mid-word and
separates ideas that belong together, which degrades retrieval quality.
Our strategy preserves meaning at every level:

1. **Paragraph boundaries** (\\n\\n) — the strongest semantic signal in
   plain text. A paragraph usually contains one idea; keeping it intact
   means the embedding captures that idea faithfully.

2. **Sentence boundaries** — when a paragraph exceeds the chunk size, we
   split on sentence-ending punctuation (.!?) followed by whitespace.
   Sentences are the smallest unit that carries a complete thought, so
   this is the next-best split point.

3. **Hard character limit** — a last resort for extremely long sentences
   (e.g. legal run-ons or code blocks). We hard-split at `chunk_size`
   with overlap to avoid losing context at the boundary.

Overlap
-------
Between consecutive chunks we carry over the tail of the previous chunk
(controlled by `overlap`). This ensures that context which straddles a
split boundary appears in both chunks, improving retrieval recall for
queries that happen to target that boundary region.

Public API
----------
chunk_document(pages, chunk_size, overlap) -> list[dict]
    Takes the page list from document_loader and returns chunks with
    metadata.
"""

from __future__ import annotations

import re


# ── Sentence-boundary regex ──────────────────────────────────────────────────
# Matches a sentence-ending punctuation mark followed by whitespace.
# We use a lookbehind so the punctuation stays attached to the sentence it
# ends, rather than becoming the start of the next chunk.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ── Public API ───────────────────────────────────────────────────────────────

def chunk_document(
    pages: list[dict],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """Split a document's pages into overlapping, metadata-tagged chunks.

    Parameters
    ----------
    pages : list[dict]
        Output of ``document_loader.load_document`` — each dict has
        ``"text"`` and ``"metadata"`` (with ``"source"`` and ``"page"``).
    chunk_size : int
        Target maximum character length per chunk (default 800).
    overlap : int
        Number of characters to carry over between consecutive chunks
        (default 150). Must be less than ``chunk_size``.

    Returns
    -------
    list[dict]
        Each chunk dict::

            {
                "text": str,
                "metadata": {
                    "source": str,
                    "chunk_index": int,
                    "page_number": int | None
                }
            }

    Raises
    ------
    ValueError
        If ``overlap >= chunk_size`` or ``chunk_size < 1``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    source: str = pages[0]["metadata"]["source"] if pages else "unknown"
    all_chunks: list[dict] = []
    chunk_index: int = 0

    for page in pages:
        text: str = page["text"]
        page_number: int | None = page["metadata"].get("page")

        # Tier 1: split on paragraph boundaries (double newline).
        paragraphs: list[str] = _split_paragraphs(text)

        # Accumulate paragraphs into chunks up to chunk_size.
        raw_segments: list[str] = _merge_segments(paragraphs, chunk_size)

        # Tier 2: any segment still over chunk_size gets sentence-split.
        sentence_segments: list[str] = []
        for segment in raw_segments:
            if len(segment) <= chunk_size:
                sentence_segments.append(segment)
            else:
                sentence_segments.extend(
                    _split_by_sentences(segment, chunk_size)
                )

        # Tier 3: any segment STILL over chunk_size gets hard-split.
        final_segments: list[str] = []
        for segment in sentence_segments:
            if len(segment) <= chunk_size:
                final_segments.append(segment)
            else:
                final_segments.extend(
                    _hard_split(segment, chunk_size)
                )

        # Apply overlap between consecutive segments.
        overlapped: list[str] = _apply_overlap(final_segments, overlap)

        # Build chunk dicts with metadata.
        for segment_text in overlapped:
            all_chunks.append({
                "text": segment_text,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                },
            })
            chunk_index += 1

    return all_chunks


# ── Internal helpers ─────────────────────────────────────────────────────────

def _split_paragraphs(text: str) -> list[str]:
    """Split text on double-newline boundaries, discarding blanks.

    A paragraph break is the strongest semantic boundary in plain text,
    so we split here first before considering finer-grained options.
    """
    raw = text.split("\n\n")
    return [p.strip() for p in raw if p.strip()]


def _merge_segments(
    pieces: list[str],
    max_length: int,
) -> list[str]:
    """Greedily merge consecutive pieces until adding the next would exceed max_length.

    This avoids creating tiny chunks when several short paragraphs appear
    in a row — small chunks embed poorly because they lack context.
    """
    merged: list[str] = []
    current: str = ""

    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_length:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece

    if current:
        merged.append(current)

    return merged


def _split_by_sentences(text: str, max_length: int) -> list[str]:
    """Split text on sentence boundaries, merging short sentences together.

    Sentences are the smallest unit that carries a complete thought. By
    splitting here (rather than at arbitrary character offsets) we keep
    each chunk semantically coherent, which produces better embeddings.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return _merge_segments(sentences, max_length)


def _hard_split(text: str, max_length: int) -> list[str]:
    """Last-resort character-level split for extremely long sentences.

    Tries to break on the last whitespace before the limit so we don't
    cut words in half. Falls back to a raw character cut only if no
    whitespace exists in the entire chunk (e.g. a very long URL).
    """
    segments: list[str] = []
    remaining = text

    while len(remaining) > max_length:
        # Find the last whitespace before the limit.
        split_point = remaining.rfind(" ", 0, max_length)
        if split_point == -1:
            # No whitespace at all — forced to cut mid-token.
            split_point = max_length

        segments.append(remaining[:split_point].strip())
        remaining = remaining[split_point:].strip()

    if remaining:
        segments.append(remaining)

    return segments


def _apply_overlap(segments: list[str], overlap: int) -> list[str]:
    """Prepend the tail of the previous chunk to each subsequent chunk.

    Overlap ensures that context straddling a chunk boundary appears in
    both chunks. A query whose answer spans two paragraphs will match
    at least one of the overlapping chunks with high similarity.

    We overlap by characters (not tokens) because the chunking contract
    is character-based. The overlap is taken from the *end* of the
    previous segment and prepended to the *start* of the next.
    """
    if overlap <= 0 or len(segments) <= 1:
        return segments

    result: list[str] = [segments[0]]

    for i in range(1, len(segments)):
        prev_tail = segments[i - 1][-overlap:]
        # Avoid duplicating text if the segment already starts with the tail
        # (can happen when paragraphs are very short).
        if segments[i].startswith(prev_tail):
            result.append(segments[i])
        else:
            result.append(f"{prev_tail} {segments[i]}")

    return result
