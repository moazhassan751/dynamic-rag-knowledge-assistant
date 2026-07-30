"""
Retrieval & Re-ranking — from scratch, no LangChain Retriever.

This module implements the full retrieval pipeline:
1. Embed the user's query with the same model used for indexing.
2. Query Pinecone for the top-k most similar chunks.
3. Re-rank the results using a hybrid scoring function.
4. Return the re-ranked context to the generator.

Why re-rank after vector search?
---------------------------------
Pure cosine similarity excels at capturing semantic meaning but can miss
exact keyword matches that are strong relevance signals. For example:

- A query about "Python GIL" should strongly prefer chunks that literally
  contain "GIL" — even if another chunk about "concurrency" is
  semantically close but never mentions the term.

- Technical acronyms, proper nouns, and domain jargon are often poorly
  represented in general-purpose embedding spaces.

Our hybrid re-ranking blends cosine similarity (70%) with keyword overlap
(30%) to capture both semantic and lexical relevance. The weights are
chosen conservatively — cosine still dominates because the embedding model
is the primary retrieval signal, but keyword overlap provides a meaningful
boost for precise terminology matches.

Public API
----------
retrieve(query, embedder, vector_store, top_k) -> list[dict]
    End-to-end retrieval: embed → search → re-rank.

rerank(query, results) -> list[dict]
    Standalone re-ranking function (visible, testable, separate).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.embeddings import EmbeddingProvider
    from src.vector_store import PineconeVectorStore


# ── Common English stopwords ─────────────────────────────────────────────────
# A minimal set of high-frequency words that carry little semantic meaning.
# We strip these during keyword overlap scoring to avoid inflating the score
# with ubiquitous words like "the", "is", "a". This list is intentionally
# small — aggressive stopword removal can hurt queries where a stopword is
# actually significant (e.g. "To be or not to be").
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "between", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "so", "if", "then", "than",
    "what", "which", "who", "whom", "how", "when", "where", "why",
})

# Regex to extract word tokens (alphanumeric sequences).
_TOKENIZE_RE = re.compile(r"[a-z0-9]+")


# ── Public API ───────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    embedder: "EmbeddingProvider",
    vector_store: "PineconeVectorStore",
    top_k: int = 5,
) -> list[dict]:
    """Run the full retrieval pipeline: embed → search → re-rank.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    embedder : EmbeddingProvider
        The embedding model (must be the same one used during indexing).
    vector_store : PineconeVectorStore
        The Pinecone wrapper to search.
    top_k : int
        Number of results to retrieve from Pinecone (default 5).

    Returns
    -------
    list[dict]
        Re-ranked results, each with keys::

            {
                "id": str,
                "text": str,
                "source": str,
                "chunk_index": int,
                "page_number": int | None,
                "cosine_score": float,
                "keyword_score": float,
                "final_score": float,
            }
    """
    # Step 1: Embed the query.
    query_vector: list[float] = embedder.embed_query(query)

    # Step 2: Vector search.
    raw_results: list[dict] = vector_store.query(
        vector=query_vector,
        top_k=top_k,
    )

    if not raw_results:
        return []

    # Step 3: Re-rank.
    return rerank(query, raw_results)


def rerank(query: str, results: list[dict]) -> list[dict]:
    """Re-rank retrieval results using hybrid cosine + keyword scoring.

    This function is deliberately kept separate from ``retrieve()`` so it
    can be tested, swapped, or extended independently. For example, you
    could replace this with a cross-encoder model without touching the
    retrieval logic.

    Scoring formula
    ---------------
    ``final_score = 0.7 * cosine_score + 0.3 * keyword_overlap_score``

    - **cosine_score** (0–1): The cosine similarity from Pinecone.
    - **keyword_overlap_score** (0–1): Jaccard similarity between the
      tokenised query and chunk text (after lowercasing and stopword
      removal).

    The 70/30 weighting reflects our belief that semantic similarity
    (from a well-trained embedding model) is the primary relevance
    signal, while keyword overlap provides a correction factor for
    terminology precision. These weights could be tuned empirically
    on a labelled relevance dataset if one were available.

    Parameters
    ----------
    query : str
        The original user query.
    results : list[dict]
        Raw results from Pinecone, each must have ``"score"`` and
        ``"text"`` keys.

    Returns
    -------
    list[dict]
        Results sorted by ``final_score`` descending, with scoring
        fields attached.
    """
    query_tokens = _tokenize(query)

    scored: list[dict] = []
    for result in results:
        chunk_tokens = _tokenize(result.get("text", ""))
        kw_score = _jaccard_similarity(query_tokens, chunk_tokens)
        cosine_score = result.get("score", 0.0)

        final_score = 0.7 * cosine_score + 0.3 * kw_score

        scored.append({
            "id": result.get("id", ""),
            "text": result.get("text", ""),
            "source": result.get("source", ""),
            "chunk_index": result.get("chunk_index", -1),
            "page_number": result.get("page_number"),
            "cosine_score": round(cosine_score, 4),
            "keyword_score": round(kw_score, 4),
            "final_score": round(final_score, 4),
        })

    # Sort descending by final score.
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored


# ── Internal helpers ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lowercase, extract word tokens, and remove stopwords.

    Returns a set (not a list) because Jaccard similarity operates on
    sets — we care about *which* terms appear, not how many times.
    """
    tokens = set(_TOKENIZE_RE.findall(text.lower()))
    return tokens - _STOPWORDS


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity: |A ∩ B| / |A ∪ B|.

    Returns 0.0 if both sets are empty (avoids division by zero).

    Jaccard is a natural choice for keyword overlap because it:
    - Is bounded [0, 1], making it easy to blend with cosine scores
    - Penalises both missing relevant terms and adding irrelevant ones
    - Is symmetric: query↔chunk overlap is the same in both directions
    """
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union
