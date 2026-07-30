"""
Embedding Providers — direct SDK calls, no LangChain wrappers.

This module defines a Protocol-based interface (``EmbeddingProvider``) and
two concrete implementations:

- **HuggingFaceEmbedder**: loads a local ``sentence-transformers`` model and
  calls ``model.encode()`` directly. Default model: all-MiniLM-L6-v2 (384-dim).
- **OpenAIEmbedder**: calls the OpenAI embeddings API via the ``openai`` SDK.
  Default model: text-embedding-3-small (1536-dim).

Why a Protocol instead of an ABC?
---------------------------------
A Protocol lets us enforce the contract at type-check time (mypy / pyright)
without forcing every embedder to inherit from a base class. This is more
Pythonic and keeps the implementations decoupled — important if someone wants
to add a third provider later.

Why batch embedding matters
---------------------------
Embedding one text at a time incurs per-request overhead (HTTP round-trip
for OpenAI, GPU kernel launch for local models). Batching amortises this
cost and can be 10-50x faster for large document sets.

Public API
----------
get_embedder(provider, settings) -> EmbeddingProvider
    Factory function that returns the correct embedder based on config.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Settings


# ── Provider Protocol ────────────────────────────────────────────────────────

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract that every embedding provider must satisfy."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return their vector representations."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Convenience wrapper around embed_batch."""
        ...

    @property
    def dimension(self) -> int:
        """The dimensionality of vectors this provider produces."""
        ...


# ── HuggingFace (Local) ─────────────────────────────────────────────────────

class HuggingFaceEmbedder:
    """Embeds text using a local sentence-transformers model.

    Uses ``SentenceTransformer.encode()`` directly — no LangChain
    ``HuggingFaceEmbeddings`` wrapper. Batching is handled by the
    underlying model; we just pass the full list and a batch_size hint.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self._dimension: int = self._model.get_sentence_embedding_dimension()

    # ── Public interface ─────────────────────────────────────────────────

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single batched call.

        Parameters
        ----------
        texts : list[str]
            The texts to embed. Can be any size — the model handles
            internal batching via ``batch_size``.

        Returns
        -------
        list[list[float]]
            One vector per input text, each of length ``self.dimension``.
        """
        if not texts:
            return []

        # encode() returns a numpy ndarray of shape (n, dim).
        # batch_size=64 balances throughput vs. memory on typical hardware.
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Wraps embed_batch for convenience. Kept as a separate method
        because some providers (e.g. OpenAI) distinguish between
        document and query embedding calls.
        """
        return self.embed_batch([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


# ── OpenAI (API) ─────────────────────────────────────────────────────────────

class OpenAIEmbedder:
    """Embeds text via the OpenAI embeddings API.

    Uses the ``openai`` SDK directly — no LangChain wrapper. The API
    natively supports batching (multiple inputs per request), which we
    leverage to minimise HTTP round-trips.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
    ) -> None:
        # Import here to avoid requiring the openai package when using
        # the HuggingFace provider.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model_name
        # text-embedding-3-small outputs 1536 dimensions by default.
        self._dimension: int = 1536

    # ── Public interface ─────────────────────────────────────────────────

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        The OpenAI API accepts up to 2048 inputs per request. For very
        large batches we chunk into sub-batches of 2048 to stay within
        limits while still minimising round-trips.

        Parameters
        ----------
        texts : list[str]
            The texts to embed.

        Returns
        -------
        list[list[float]]
            One vector per input text, each of length ``self.dimension``.

        Raises
        ------
        openai.APIError
            If the API call fails (rate limit, auth, etc.).
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_limit = 2048  # OpenAI's per-request input limit

        for start in range(0, len(texts), batch_limit):
            batch = texts[start : start + batch_limit]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            # Response data is ordered by index, but we sort to be safe.
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([item.embedding for item in sorted_data])

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_batch([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


# ── Factory ──────────────────────────────────────────────────────────────────

def get_embedder(settings: Settings) -> EmbeddingProvider:
    """Instantiate the correct embedder based on application config.

    Parameters
    ----------
    settings : Settings
        The application settings (from ``config.load_settings()``).

    Returns
    -------
    EmbeddingProvider
        A ready-to-use embedder matching the configured provider.

    Raises
    ------
    ValueError
        If the configured provider is not recognised.
    """
    provider = settings.embedding_provider

    if provider == "huggingface":
        return HuggingFaceEmbedder(model_name=settings.hf_embedding_model)

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER='openai'."
            )
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model_name=settings.openai_embedding_model,
        )

    raise ValueError(
        f"Unknown embedding provider '{provider}'. "
        "Supported: 'huggingface', 'openai'."
    )
