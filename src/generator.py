"""
Answer Generation — direct Groq/OpenAI API calls, no LangChain chains.

This module constructs a hand-written prompt from the retrieved context
and the user's question, then sends it to the configured LLM provider
for completion. No LangChain Chain, PromptTemplate, or LCEL is used.

Prompt design rationale
-----------------------
The system prompt explicitly instructs the LLM to:
1. Answer ONLY from provided context — prevents hallucination.
2. Cite source document names — gives the user traceability.
3. Say "I cannot find the answer" when context is insufficient — honest
   uncertainty is better than a confident wrong answer.

We avoid complex prompt engineering (few-shot examples, chain-of-thought)
in the base prompt because the goal is grounded retrieval, not reasoning.
If the context contains the answer, a well-instructed model will find it;
if it doesn't, we want the model to say so rather than guess.

Public API
----------
get_generator(settings) -> Generator (Protocol-compatible object)
generate_answer(query, context_chunks, generator) -> dict
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from config import Settings
from src.logger import setup_logger

logger = setup_logger(__name__)


# ── Prompt Template ──────────────────────────────────────────────────────────
# Hand-written, no LangChain PromptTemplate.

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the user's question using ONLY "
    "the provided context. Follow these rules strictly:\n"
    "\n"
    "1. Base your answer entirely on the context below. Do NOT use prior "
    "knowledge or make assumptions beyond what the context states.\n"
    "2. If the context does not contain enough information to answer the "
    'question, respond with: "I cannot find the answer in the provided '
    'documents."\n'
    "3. When possible, cite the source document name in your answer "
    '(e.g. "According to report.pdf...").\n'
    "4. Be concise but thorough. Include relevant details from the context.\n"
)


def _build_context_block(context_chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is labelled with its source filename and (if available)
    page number. Numbering helps the LLM reference specific chunks and
    gives the user a way to trace claims back to their origin.
    """
    parts: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page_number")
        header = f"[{i}] Source: {source}"
        if page is not None:
            header += f" (Page {page})"

        parts.append(f"{header}\n{chunk.get('text', '')}")

    return "\n\n---\n\n".join(parts)


def _build_user_message(query: str, context_chunks: list[dict]) -> str:
    """Assemble the full user message with context and question."""
    context_block = _build_context_block(context_chunks)
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


# ── Generator Protocol ───────────────────────────────────────────────────────

@runtime_checkable
class Generator(Protocol):
    """Contract for LLM generation providers."""

    def generate(self, query: str, context_chunks: list[dict]) -> dict[str, Any]:
        """Generate an answer from query + context.

        Returns
        -------
        dict with keys: "answer" (str), "model" (str), "usage" (dict)
        """
        ...


# ── Groq Provider ────────────────────────────────────────────────────────────

class GroqGenerator:
    """Generate answers via the Groq API (LLaMA models).

    Uses the ``groq`` SDK directly — the API is OpenAI-compatible but
    runs on Groq's LPU hardware for extremely low-latency inference.
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._model = model

    def generate(self, query: str, context_chunks: list[dict]) -> dict[str, Any]:
        """Send the prompt to Groq and return the structured response.

        Parameters
        ----------
        query : str
            The user's question.
        context_chunks : list[dict]
            Retrieved and re-ranked context chunks.

        Returns
        -------
        dict
            ``{"answer": str, "model": str, "usage": dict}``

        Raises
        ------
        groq.APIError
            If the Groq API call fails.
        """
        user_message = _build_user_message(query, context_chunks)

        logger.info(f"Generating answer with Groq model '{self._model}'")
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,  # Low temperature for factual, grounded answers
                max_tokens=1024,
            )
            logger.debug("Groq generation complete.")
        except Exception as exc:
            logger.error(f"Groq API call failed: {exc}", exc_info=True)
            raise

        choice = response.choices[0]
        usage = response.usage

        return {
            "answer": choice.message.content.strip(),
            "model": self._model,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }


# ── OpenAI Provider ──────────────────────────────────────────────────────────

class OpenAIGenerator:
    """Generate answers via the OpenAI Chat Completions API.

    Used as the swappable alternative to Groq. Same prompt, same
    response structure — only the underlying model differs.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, query: str, context_chunks: list[dict]) -> dict[str, Any]:
        """Send the prompt to OpenAI and return the structured response.

        Parameters
        ----------
        query : str
            The user's question.
        context_chunks : list[dict]
            Retrieved and re-ranked context chunks.

        Returns
        -------
        dict
            ``{"answer": str, "model": str, "usage": dict}``

        Raises
        ------
        openai.APIError
            If the OpenAI API call fails.
        """
        user_message = _build_user_message(query, context_chunks)

        logger.info(f"Generating answer with OpenAI model '{self._model}'")
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            logger.debug("OpenAI generation complete.")
        except Exception as exc:
            logger.error(f"OpenAI API call failed: {exc}", exc_info=True)
            raise

        choice = response.choices[0]
        usage = response.usage

        return {
            "answer": choice.message.content.strip(),
            "model": self._model,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }


# ── Factory ──────────────────────────────────────────────────────────────────

def get_generator(settings: Settings) -> Generator:
    """Instantiate the correct LLM generator based on config.

    Parameters
    ----------
    settings : Settings
        Application settings from ``config.load_settings()``.

    Returns
    -------
    Generator
        A ready-to-use generator matching the configured LLM provider.

    Raises
    ------
    ValueError
        If the configured provider is not recognised.
    """
    provider = settings.llm_provider

    if provider == "groq":
        return GroqGenerator(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER='openai'."
            )
        return OpenAIGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )

    raise ValueError(
        f"Unknown LLM provider '{provider}'. Supported: 'groq', 'openai'."
    )
