"""
Pinecone Vector Store Wrapper — raw ``pinecone`` client, no LangChain VectorStore.

This module owns all communication with Pinecone: creating indexes,
upserting vectors, querying by similarity, and deleting vectors by source.

Why wrap the raw client instead of using LangChain's VectorStore?
-----------------------------------------------------------------
The project requirement is to demonstrate understanding of every layer.
By calling the Pinecone SDK directly we control:
- How IDs are generated (deterministic, source-based)
- How metadata is structured and filtered
- How batching works during upserts (100 vectors per batch to stay under
  Pinecone's 2 MB payload limit)
- How deletion is verified (post-delete query to confirm zero matches)

Dual-index strategy
-------------------
Pinecone fixes the vector dimension at index creation time. Since we support
both 384-dim (HuggingFace) and 1536-dim (OpenAI) embeddings, we maintain two
separate indexes. ``config.Settings.active_pinecone_index`` selects the right
one based on the currently configured embedding provider.

Public API
----------
PineconeVectorStore(settings)
    .upsert_chunks(chunks, embeddings) -> int
    .query(vector, top_k) -> list[dict]
    .delete_by_source(source) -> int
    .list_sources() -> list[str]
    .get_index_stats() -> dict
"""

from __future__ import annotations

import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from config import Settings
from src.logger import setup_logger

logger = setup_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
_UPSERT_BATCH_SIZE = 100  # Pinecone recommends ≤100 vectors per upsert call
_DELETE_VERIFY_DELAY = 1.0  # Seconds to wait before verifying deletion


class PineconeVectorStore:
    """Manages a single Pinecone serverless index for document vectors.

    The index is auto-created if it doesn't exist (idempotent). All
    vectors carry metadata ``{source, chunk_index, page_number, text}``
    so we can filter and display context downstream.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.active_pinecone_index
        self._dimension = settings.active_embedding_dimension

        self._ensure_index_exists()
        self._index = self._pc.Index(self._index_name)

    # ── Index lifecycle ──────────────────────────────────────────────────

    def _ensure_index_exists(self) -> None:
        """Create the serverless index if it doesn't already exist.

        Uses cosine similarity (the standard for sentence-transformer
        models) and the cloud/region from config. This is idempotent —
        if the index already exists Pinecone simply returns it.
        """
        existing = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name in existing:
            logger.info("Pinecone index '%s' already exists.", self._index_name)
            return

        logger.info(
            "Creating Pinecone index '%s' (dim=%d, metric=%s).",
            self._index_name,
            self._dimension,
            self._settings.pinecone_metric,
        )
        self._pc.create_index(
            name=self._index_name,
            dimension=self._dimension,
            metric=self._settings.pinecone_metric,
            spec=ServerlessSpec(
                cloud=self._settings.pinecone_cloud,
                region=self._settings.pinecone_region,
            ),
        )
        # Wait for index to become ready.
        self._wait_for_index_ready()

    def _wait_for_index_ready(self, timeout: int = 120) -> None:
        """Block until the index status is 'Ready', up to ``timeout`` seconds."""
        start = time.time()
        while time.time() - start < timeout:
            desc = self._pc.describe_index(self._index_name)
            if desc.status.get("ready", False):
                logger.info("Index '%s' is ready.", self._index_name)
                return
            time.sleep(2)
        logger.warning(
            "Index '%s' did not become ready within %ds.", self._index_name, timeout
        )

    # ── Upsert ───────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """Upsert chunk vectors with metadata into Pinecone.

        Parameters
        ----------
        chunks : list[dict]
            Output of ``chunking.chunk_document``. Each dict must have
            ``"text"`` and ``"metadata"`` with ``"source"``,
            ``"chunk_index"``, and ``"page_number"``.
        embeddings : list[list[float]]
            Parallel list of embedding vectors (same length as chunks).

        Returns
        -------
        int
            Number of vectors upserted.

        Raises
        ------
        ValueError
            If chunks and embeddings have mismatched lengths.
        pinecone.exceptions.PineconeApiException
            If the Pinecone API call fails.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        vectors: list[tuple[str, list[float], dict[str, Any]]] = []
        for chunk, embedding in zip(chunks, embeddings):
            meta = chunk["metadata"]
            # Deterministic ID: source filename + chunk index.
            # This makes re-indexing idempotent — upserting the same document
            # again overwrites rather than duplicates.
            vec_id = f"{meta['source']}_{meta['chunk_index']}"
            vectors.append((
                vec_id,
                embedding,
                {
                    "source": meta["source"],
                    "chunk_index": meta["chunk_index"],
                    "page_number": meta.get("page_number"),
                    "text": chunk["text"],
                },
            ))

        # Batch upserts to stay within Pinecone's payload limits.
        upserted = 0
        for start in range(0, len(vectors), _UPSERT_BATCH_SIZE):
            batch = vectors[start : start + _UPSERT_BATCH_SIZE]
            self._index.upsert(vectors=batch)
            upserted += len(batch)
            logger.debug("Upserted batch of %d vectors.", len(batch))

        logger.info(
            "Upserted %d vectors for source '%s'.",
            upserted,
            chunks[0]["metadata"]["source"] if chunks else "unknown",
        )
        return upserted

    # ── Query ────────────────────────────────────────────────────────────

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[dict]:
        """Find the top-k most similar vectors.

        Parameters
        ----------
        vector : list[float]
            The query embedding.
        top_k : int
            Number of results to return (default 5).
        source_filter : str | None
            If provided, restrict results to vectors from this source file.

        Returns
        -------
        list[dict]
            Each result dict::

                {
                    "id": str,
                    "score": float,       # cosine similarity
                    "text": str,
                    "source": str,
                    "chunk_index": int,
                    "page_number": int | None
                }
        """
        query_kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True,
        }
        if source_filter:
            query_kwargs["filter"] = {"source": {"$eq": source_filter}}

        response = self._index.query(**query_kwargs)

        results: list[dict] = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append({
                "id": match.id,
                "score": float(match.score),
                "text": meta.get("text", ""),
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "page_number": meta.get("page_number"),
            })

        return results

    # ── Delete ───────────────────────────────────────────────────────────

    def delete_by_source(self, source: str) -> bool:
        """Delete ALL vectors associated with a source document.

        Uses Pinecone's metadata-filter delete to remove every vector
        where ``metadata.source == source``. After deletion, runs a
        verification query to confirm zero vectors remain — this catches
        eventual-consistency issues and gives the caller confidence that
        the knowledge base is truly clean.

        Parameters
        ----------
        source : str
            The source filename whose vectors should be deleted.

        Returns
        -------
        bool
            True if deletion was verified (zero vectors remain for this
            source), False if residual vectors were detected.
        """
        logger.info("Deleting all vectors for source '%s'.", source)

        self._index.delete(
            filter={"source": {"$eq": source}},
        )

        # ── Verify deletion ──────────────────────────────────────────────
        # Pinecone serverless is eventually consistent — wait briefly
        # then query to confirm the vectors are gone.
        time.sleep(_DELETE_VERIFY_DELAY)

        # We need a dummy vector to query. Use a zero vector — the scores
        # won't be meaningful but we only care about whether any matches
        # exist for this source filter.
        dummy_vector = [0.0] * self._dimension
        remaining = self.query(
            vector=dummy_vector,
            top_k=1,
            source_filter=source,
        )

        if remaining:
            logger.warning(
                "Verification failed: %d vectors still remain for source '%s'. "
                "This may be due to eventual consistency — retry shortly.",
                len(remaining),
                source,
            )
            return False

        logger.info("Verified: 0 vectors remain for source '%s'.", source)
        return True

    # ── Metadata queries ─────────────────────────────────────────────────

    def list_sources(self) -> list[str]:
        """Return the distinct source filenames currently in the index.

        Pinecone doesn't support SELECT DISTINCT on metadata, so we use
        a heuristic: query with a zero vector and a high top_k, then
        extract unique sources from the results. This works well for
        small-to-medium knowledge bases (< 10k vectors).

        For production scale, you'd maintain a separate source registry
        (e.g. in SQLite). For this project's scope, the heuristic is
        sufficient and avoids adding another persistence layer for
        document tracking.
        """
        dummy_vector = [0.0] * self._dimension
        response = self._index.query(
            vector=dummy_vector,
            top_k=10_000,  # Large enough to capture all vectors
            include_metadata=True,
        )

        sources: set[str] = set()
        for match in response.matches:
            if match.metadata and "source" in match.metadata:
                sources.add(match.metadata["source"])

        return sorted(sources)

    def get_index_stats(self) -> dict:
        """Return index statistics (vector count, dimension, etc.).

        Useful for the Streamlit sidebar to show the current state
        of the knowledge base.
        """
        stats = self._index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": self._dimension,
            "index_name": self._index_name,
        }
