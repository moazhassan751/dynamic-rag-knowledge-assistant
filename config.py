"""
Centralised configuration for the Dynamic RAG Knowledge Assistant.

Reads environment variables from a .env file (via python-dotenv) and exposes
them as a typed, validated Settings dataclass. Every module imports from here
rather than reading os.environ directly — this keeps API keys, model names,
and tunables in one auditable place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


# ── Settings dataclass ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """Immutable application settings derived from environment variables.

    Frozen to prevent accidental mutation at runtime — any change should go
    through the .env file and a restart.
    """

    # ── API Keys ─────────────────────────────────────────────────────────
    pinecone_api_key: str = field(repr=False)  # repr=False keeps keys out of logs
    groq_api_key: str = field(repr=False)
    openai_api_key: str = field(default="", repr=False)  # Optional

    # ── Provider Selection ───────────────────────────────────────────────
    embedding_provider: str = "huggingface"  # "huggingface" | "openai"
    llm_provider: str = "groq"               # "groq" | "openai"

    # ── Embedding Models ─────────────────────────────────────────────────
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── LLM Models ───────────────────────────────────────────────────────
    groq_model: str = "llama-3.3-70b-versatile"
    openai_model: str = "gpt-4o-mini"

    # ── Pinecone ─────────────────────────────────────────────────────────
    pinecone_index_384: str = "rag-384"
    pinecone_index_1536: str = "rag-1536"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"

    # ── Chunking Defaults ────────────────────────────────────────────────
    default_chunk_size: int = 800
    default_chunk_overlap: int = 150

    # ── Retrieval Defaults ───────────────────────────────────────────────
    default_top_k: int = 5

    # ── Evaluation ───────────────────────────────────────────────────────
    eval_db_path: str = "eval_logs.db"

    # ── Derived Properties ───────────────────────────────────────────────

    @property
    def active_pinecone_index(self) -> str:
        """Return the Pinecone index name matching the active embedding dimension."""
        if self.embedding_provider == "openai":
            return self.pinecone_index_1536
        return self.pinecone_index_384

    @property
    def active_embedding_dimension(self) -> int:
        """Return the vector dimension for the active embedding provider."""
        if self.embedding_provider == "openai":
            return 1536
        return 384


def load_settings() -> Settings:
    """Build a Settings instance from the current environment.

    Raises:
        ValueError: If a required API key is missing.
    """
    pinecone_key = os.getenv("PINECONE_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()

    # ── Validate required keys ───────────────────────────────────────────
    if not pinecone_key:
        raise ValueError(
            "PINECONE_API_KEY is required. "
            "Set it in your .env file or as an environment variable."
        )

    if llm_provider == "groq" and not groq_key:
        raise ValueError(
            "GROQ_API_KEY is required when LLM_PROVIDER='groq'. "
            "Set it in your .env file or as an environment variable."
        )

    if (llm_provider == "openai" or embedding_provider == "openai") and not openai_key:
        raise ValueError(
            "OPENAI_API_KEY is required when using OpenAI as the LLM or "
            "embedding provider. Set it in your .env file."
        )

    if embedding_provider not in ("huggingface", "openai"):
        raise ValueError(
            f"EMBEDDING_PROVIDER must be 'huggingface' or 'openai', "
            f"got '{embedding_provider}'."
        )

    if llm_provider not in ("groq", "openai"):
        raise ValueError(
            f"LLM_PROVIDER must be 'groq' or 'openai', got '{llm_provider}'."
        )

    return Settings(
        pinecone_api_key=pinecone_key,
        groq_api_key=groq_key,
        openai_api_key=openai_key,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        pinecone_index_384=os.getenv("PINECONE_INDEX_384", "rag-384"),
        pinecone_index_1536=os.getenv("PINECONE_INDEX_1536", "rag-1536"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        eval_db_path=os.getenv("EVAL_DB_PATH", "eval_logs.db"),
    )
