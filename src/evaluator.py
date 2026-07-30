"""
Evaluation Layer — fully custom, no RAGAS or pre-built eval framework.

This module measures three dimensions of RAG performance per query and
persists the results to a SQLite database for dashboard visualisation:

1. **Retrieval Quality** — how relevant are the retrieved chunks?
   Measured as the average cosine similarity score of the top-k results.

2. **Latency Breakdown** — where is time spent?
   Separately timed: embedding, retrieval, generation, and total.

3. **Faithfulness** — is the answer actually supported by the context?
   A second LLM call acts as a "judge", scoring whether the generated
   answer is grounded in the retrieved context or hallucinates beyond it.

Why build this from scratch?
-----------------------------
Pre-built frameworks like RAGAS abstract away the evaluation logic,
making it hard to understand what's actually being measured. By
implementing each metric ourselves, we:
- Control exactly what each score means
- Can explain the methodology in detail during review
- Avoid pulling in heavy dependencies for a few simple calculations

SQLite persistence
------------------
Metrics are stored in ``eval_logs.db`` so the Streamlit dashboard can
show historical trends, not just the current query. SQLite is chosen
because it's zero-config, file-based (easy to deploy), and handles
concurrent reads from Streamlit's multi-threaded model.

Public API
----------
LatencyTracker — context manager for timing pipeline stages
score_retrieval_relevance(results) -> float
score_faithfulness(query, context, answer, settings) -> dict
log_evaluation(metrics) -> None
get_recent_evaluations(limit) -> list[dict]
compare_states(metrics_before, metrics_after) -> dict
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from config import Settings
from src.logger import setup_logger

logger = setup_logger(__name__)

# ── Database Schema ──────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS eval_logs (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                   TEXT    NOT NULL,
    query                       TEXT    NOT NULL,
    answer                      TEXT    NOT NULL,
    retrieval_relevance         REAL,
    faithfulness_score          REAL,
    faithfulness_justification  TEXT,
    embed_ms                    REAL,
    retrieve_ms                 REAL,
    generate_ms                 REAL,
    total_ms                    REAL,
    num_documents               INTEGER,
    document_sources            TEXT,
    top_k                       INTEGER,
    model_used                  TEXT
);
"""


# ── Latency Tracking ────────────────────────────────────────────────────────

class LatencyTracker:
    """Accumulates timing measurements for each pipeline stage.

    Usage::

        tracker = LatencyTracker()

        with tracker.track("embed"):
            embedding = embedder.embed_query(query)

        with tracker.track("retrieve"):
            results = vector_store.query(embedding, top_k=5)

        with tracker.track("generate"):
            answer = generator.generate(query, results)

        print(tracker.get_breakdown())
        # {"embed_ms": 12.3, "retrieve_ms": 45.6, "generate_ms": 234.5, "total_ms": 292.4}
    """

    def __init__(self) -> None:
        self._timings: dict[str, float] = {}
        self._start_time: float = time.perf_counter()

    @contextmanager
    def track(self, stage: str) -> Generator[None, None, None]:
        """Time a named pipeline stage.

        Parameters
        ----------
        stage : str
            Name of the stage (e.g. "embed", "retrieve", "generate").
            The resulting key in ``get_breakdown()`` will be ``{stage}_ms``.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._timings[f"{stage}_ms"] = round(elapsed_ms, 2)

    def get_breakdown(self) -> dict[str, float]:
        """Return all recorded timings plus a ``total_ms`` field.

        ``total_ms`` is the wall-clock time from when the tracker was
        created, NOT the sum of individual stages. This captures any
        overhead between stages (e.g. re-ranking, data marshalling).
        """
        total = (time.perf_counter() - self._start_time) * 1000
        return {
            **self._timings,
            "total_ms": round(total, 2),
        }


# ── Retrieval Quality ───────────────────────────────────────────────────────

def score_retrieval_relevance(results: list[dict]) -> float:
    """Score retrieval quality as the average cosine similarity of top-k results.

    This is a simple but effective proxy for retrieval relevance:
    - High average score → the retrieved chunks are semantically close
      to the query, likely containing relevant information.
    - Low average score → the knowledge base may not cover the topic,
      or the embedding model is struggling with the domain vocabulary.

    A more sophisticated approach would use an LLM-as-judge to rate
    each chunk's relevance, but the average cosine score is cheap,
    deterministic, and correlates well with human relevance judgments
    for well-trained embedding models.

    Parameters
    ----------
    results : list[dict]
        Retrieval results, each must have a ``"cosine_score"`` or
        ``"score"`` key.

    Returns
    -------
    float
        Average similarity score in [0, 1], or 0.0 if no results.
    """
    if not results:
        return 0.0

    scores = [
        r.get("cosine_score", r.get("score", 0.0))
        for r in results
    ]
    return round(sum(scores) / len(scores), 4)


# ── Faithfulness Scoring ────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = (
    "You are an impartial evaluator. Your task is to assess whether an AI "
    "assistant's answer is faithfully supported by the provided context.\n"
    "\n"
    "Score the answer on a scale of 1 to 5:\n"
    "  1 = The answer is completely unsupported or contradicts the context.\n"
    "  2 = The answer contains significant claims not found in the context.\n"
    "  3 = The answer is partially supported but includes some unsupported claims.\n"
    "  4 = The answer is mostly supported with minor extrapolations.\n"
    "  5 = The answer is fully supported by the context with no hallucination.\n"
    "\n"
    "Respond with EXACTLY this format (no extra text):\n"
    "Score: <number>\n"
    "Justification: <one sentence explaining your rating>\n"
)


def score_faithfulness(
    query: str,
    context_chunks: list[dict],
    answer: str,
    settings: Settings,
) -> dict[str, Any]:
    """Score whether the generated answer is supported by the retrieved context.

    Makes a second LLM call using the same configured provider, but with
    a "judge" system prompt that evaluates faithfulness rather than
    answering the question. This is a lightweight version of the
    "LLM-as-judge" pattern.

    Why a second LLM call?
    ----------------------
    Self-evaluation (asking the same model that generated the answer to
    judge it) has known biases, but it's a practical and widely-used
    approach when human evaluation isn't available. The key is that the
    judge prompt is fundamentally different from the generation prompt —
    it doesn't ask for an answer, it asks for a rating. This forces the
    model into an evaluative mode rather than a generative one.

    Parameters
    ----------
    query : str
        The original user question.
    context_chunks : list[dict]
        The retrieved context passed to the generator.
    answer : str
        The generated answer to evaluate.
    settings : Settings
        Application settings (used to initialise the judge LLM).

    Returns
    -------
    dict
        ``{"score": float, "justification": str}``
        Score is 1-5, or 0.0 if parsing fails.
    """
    # Format the context for the judge.
    context_text = "\n\n".join(
        f"[{i+1}] {chunk.get('text', '')}"
        for i, chunk in enumerate(context_chunks)
    )

    judge_message = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer to evaluate:\n{answer}\n"
    )

    # Use the same LLM provider for judging.
    try:
        if settings.llm_provider == "groq":
            from groq import Groq
            client = Groq(api_key=settings.groq_api_key)
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": _FAITHFULNESS_PROMPT},
                    {"role": "user", "content": judge_message},
                ],
                temperature=0.0,  # Deterministic for evaluation
                max_tokens=150,
            )
        else:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": _FAITHFULNESS_PROMPT},
                    {"role": "user", "content": judge_message},
                ],
                temperature=0.0,
                max_tokens=150,
            )

        judge_output = response.choices[0].message.content.strip()
        parsed = _parse_faithfulness_response(judge_output)
        logger.info(f"Faithfulness score: {parsed['score']}/5.0")
        return parsed

    except Exception as exc:
        logger.error("Faithfulness scoring failed: %s", exc)
        return {"score": 0.0, "justification": f"Evaluation failed: {exc}"}


def _parse_faithfulness_response(text: str) -> dict[str, Any]:
    """Extract the numeric score and justification from the judge's response.

    Handles minor formatting variations (e.g. "Score: 4/5", "Score: 4.0").
    Falls back to 0.0 if parsing fails entirely.
    """
    score = 0.0
    justification = text  # Default to full text if parsing fails

    # Try to extract "Score: X"
    score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if score_match:
        raw_score = float(score_match.group(1))
        score = min(max(raw_score, 1.0), 5.0)  # Clamp to [1, 5]

    # Try to extract "Justification: ..."
    just_match = re.search(r"Justification:\s*(.+)", text, re.IGNORECASE)
    if just_match:
        justification = just_match.group(1).strip()

    return {"score": score, "justification": justification}


# ── Before/After Comparison ─────────────────────────────────────────────────

def compare_states(
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
) -> dict[str, Any]:
    """Compare evaluation metrics across two knowledge-base states.

    Useful for assessing the impact of adding or deleting a document.
    For example: "After adding whitepaper.pdf, retrieval relevance for
    this query improved from 0.62 to 0.81."

    Parameters
    ----------
    metrics_before : dict
        Evaluation metrics from a query before the KB change.
    metrics_after : dict
        Evaluation metrics from the same query after the KB change.

    Returns
    -------
    dict
        Delta values for each comparable metric, plus a human-readable
        summary.
    """
    comparable_keys = [
        "retrieval_relevance",
        "faithfulness_score",
        "embed_ms",
        "retrieve_ms",
        "generate_ms",
        "total_ms",
    ]

    deltas: dict[str, Any] = {}
    for key in comparable_keys:
        before_val = metrics_before.get(key, 0.0) or 0.0
        after_val = metrics_after.get(key, 0.0) or 0.0
        delta = round(after_val - before_val, 4)
        deltas[key] = {
            "before": before_val,
            "after": after_val,
            "delta": delta,
            "improved": (
                delta > 0 if key in ("retrieval_relevance", "faithfulness_score")
                else delta < 0  # Lower latency = improvement
            ),
        }

    return deltas


# ── SQLite Persistence ──────────────────────────────────────────────────────

def _get_db_connection(db_path: str) -> sqlite3.Connection:
    """Open (or create) the evaluation database and ensure the schema exists."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Dict-like access to rows
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def log_evaluation(
    metrics: dict[str, Any],
    db_path: str = "eval_logs.db",
) -> None:
    """Persist a single evaluation record to the SQLite database.

    Parameters
    ----------
    metrics : dict
        Must contain keys matching the table columns. Extra keys are
        silently ignored.
    db_path : str
        Path to the SQLite database file (created if it doesn't exist).
    """
    conn = _get_db_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO eval_logs (
                timestamp, query, answer,
                retrieval_relevance, faithfulness_score, faithfulness_justification,
                embed_ms, retrieve_ms, generate_ms, total_ms,
                num_documents, document_sources, top_k, model_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                metrics.get("query", ""),
                metrics.get("answer", ""),
                metrics.get("retrieval_relevance"),
                metrics.get("faithfulness_score"),
                metrics.get("faithfulness_justification", ""),
                metrics.get("embed_ms"),
                metrics.get("retrieve_ms"),
                metrics.get("generate_ms"),
                metrics.get("total_ms"),
                metrics.get("num_documents"),
                json.dumps(metrics.get("document_sources", [])),
                metrics.get("top_k"),
                metrics.get("model_used", ""),
            ),
        )
        conn.commit()
        logger.info(f"Saved evaluation record to {db_path}")
    except sqlite3.Error as exc:
        logger.error("Failed to log evaluation: %s", exc, exc_info=True)
    finally:
        conn.close()


def get_recent_evaluations(
    limit: int = 50,
    db_path: str = "eval_logs.db",
) -> list[dict]:
    """Retrieve the most recent evaluation records for the dashboard.

    Parameters
    ----------
    limit : int
        Maximum number of records to return (default 50).
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    list[dict]
        Records ordered by timestamp descending (most recent first).
        Returns an empty list if the database doesn't exist yet.
    """
    if not Path(db_path).exists():
        return []

    conn = _get_db_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM eval_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        logger.error("Failed to read evaluations: %s", exc)
        return []
    finally:
        conn.close()
