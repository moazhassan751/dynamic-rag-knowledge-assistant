"""
Dynamic RAG Knowledge Assistant — Streamlit Application.

This is the main entry point. Run with:
    streamlit run app.py

Three-tab interface:
    Chat         — Ask questions grounded in uploaded documents
    Documents    — Upload, view, and delete documents from the knowledge base
    Evaluation   — Dashboard showing retrieval quality, latency, and faithfulness

All backend logic is delegated to the modules in ``src/``. This file handles
only UI layout, Streamlit session state management, and orchestrating the
pipeline calls.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from config import load_settings, Settings
from src.chunking import chunk_document
from src.document_loader import DocumentLoadError, load_document
from src.embeddings import get_embedder, EmbeddingProvider
from src.evaluator import (
    LatencyTracker,
    compare_states,
    get_recent_evaluations,
    log_evaluation,
    score_faithfulness,
    score_retrieval_relevance,
)
from src.generator import get_generator, Generator
from src.retriever import retrieve
from src.vector_store import PineconeVectorStore

# -- Logging ------------------------------------------------------------------
from src.logger import setup_logger
logger = setup_logger(__name__)


# -- Page Config --------------------------------------------------------------

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Load Custom CSS ----------------------------------------------------------

def _load_css() -> None:
    """Load the custom stylesheet from assets/style.css."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


_load_css()


# -- Initialise Shared Resources (cached across reruns) -----------------------

@st.cache_resource(show_spinner="Loading settings...")
def init_settings() -> Settings:
    """Load and validate application settings once."""
    return load_settings()


@st.cache_resource(show_spinner="Loading embedding model...")
def init_embedder(_settings: Settings) -> EmbeddingProvider:
    """Initialise the embedding model once (heavy for HuggingFace)."""
    return get_embedder(_settings)


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def init_vector_store(_settings: Settings) -> PineconeVectorStore:
    """Connect to (or create) the Pinecone index once."""
    return PineconeVectorStore(_settings)


@st.cache_resource(show_spinner="Initialising LLM...")
def init_generator(_settings: Settings) -> Generator:
    """Initialise the LLM client once."""
    return get_generator(_settings)


# -- Session State Defaults ---------------------------------------------------

def _init_session_state() -> None:
    """Set default values for all session state keys."""
    defaults = {
        "chat_history": [],           # list of {"role": str, "content": str}
        "last_eval_metrics": None,    # dict — most recent query's metrics
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -- Sidebar ------------------------------------------------------------------

def _render_sidebar(settings: Settings, vector_store: PineconeVectorStore) -> dict:
    """Render the sidebar with config display and runtime controls.

    Returns a dict of user-adjustable parameters (chunk_size, top_k, etc.)
    that override the defaults for this session.
    """
    st.sidebar.title("RAG Knowledge Assistant")

    # -- Current Configuration ------------------------------------------------
    st.sidebar.subheader("Configuration")
    st.sidebar.markdown(
        f'Embedding &nbsp; <span class="config-value">{settings.embedding_provider}</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'LLM &nbsp; <span class="config-value">{settings.llm_provider}</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'Index &nbsp; <span class="config-value">{settings.active_pinecone_index}</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'Dimension &nbsp; <span class="config-value">{settings.active_embedding_dimension}</span>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # -- Runtime Tunables -----------------------------------------------------
    st.sidebar.subheader("Tunables")
    chunk_size = st.sidebar.slider(
        "Chunk Size (chars)", 200, 2000, settings.default_chunk_size, step=50,
        help="Target maximum characters per chunk.",
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (chars)", 0, 500, settings.default_chunk_overlap, step=25,
        help="Characters carried over between consecutive chunks.",
    )
    top_k = st.sidebar.slider(
        "Top-K Results", 1, 20, settings.default_top_k,
        help="Number of chunks retrieved per query.",
    )

    st.sidebar.markdown("---")

    # -- Index Stats ----------------------------------------------------------
    st.sidebar.subheader("Index Stats")
    try:
        stats = vector_store.get_index_stats()
        st.sidebar.metric("Total Vectors", stats["total_vector_count"])
    except Exception:
        st.sidebar.warning("Could not fetch index stats.")

    st.sidebar.markdown("---")

    # -- Clear Chat -----------------------------------------------------------
    if st.sidebar.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_eval_metrics = None
        st.rerun()

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
    }


# -- Helpers for metric badges ------------------------------------------------

def _relevance_badge(score: float) -> str:
    """Return an HTML badge for a retrieval relevance score."""
    css_class = "good" if score >= 0.4 else ("medium" if score >= 0.2 else "neutral")
    return f'<span class="metric-badge {css_class}">Relevance {score:.2f}</span>'


def _faithfulness_badge(score: float) -> str:
    """Return an HTML badge for a faithfulness score."""
    css_class = "good" if score >= 4.0 else ("medium" if score >= 2.5 else "neutral")
    return f'<span class="metric-badge {css_class}">Faithfulness {score:.1f}/5</span>'


def _latency_badge(ms: float) -> str:
    """Return an HTML badge for total latency."""
    css_class = "good" if ms < 1000 else ("medium" if ms < 3000 else "neutral")
    return f'<span class="metric-badge {css_class}">{ms:.0f} ms</span>'


# -- Tab 1: Chat -------------------------------------------------------------

def _render_chat_tab(
    settings: Settings,
    embedder: EmbeddingProvider,
    vector_store: PineconeVectorStore,
    generator: Generator,
    top_k: int,
) -> None:
    """Render the Q&A chat interface."""
    st.header("Chat with Your Documents")

    # Display chat history.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show per-query metrics as inline badges (assistant messages only).
            if message["role"] == "assistant" and "metrics" in message:
                m = message["metrics"]
                badges = (
                    _relevance_badge(m.get("retrieval_relevance", 0))
                    + _faithfulness_badge(m.get("faithfulness_score", 0))
                    + _latency_badge(m.get("total_ms", 0))
                )
                st.markdown(badges, unsafe_allow_html=True)

                with st.expander("Latency Breakdown"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Embed", f"{m.get('embed_ms', 0):.0f} ms")
                    col2.metric("Retrieve", f"{m.get('retrieve_ms', 0):.0f} ms")
                    col3.metric("Generate", f"{m.get('generate_ms', 0):.0f} ms")

    # Chat input.
    query = st.chat_input("Ask a question...")
    if query:
        # Add user message to history.
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Run the RAG pipeline.
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    metrics = _run_rag_pipeline(
                        query=query,
                        settings=settings,
                        embedder=embedder,
                        vector_store=vector_store,
                        generator=generator,
                        top_k=top_k,
                    )

                    answer = metrics["answer"]
                    st.markdown(answer)

                    # Inline metric badges.
                    badges = (
                        _relevance_badge(metrics.get("retrieval_relevance", 0))
                        + _faithfulness_badge(metrics.get("faithfulness_score", 0))
                        + _latency_badge(metrics.get("total_ms", 0))
                    )
                    st.markdown(badges, unsafe_allow_html=True)

                    with st.expander("Latency Breakdown"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Embed", f"{metrics.get('embed_ms', 0):.0f} ms")
                        col2.metric("Retrieve", f"{metrics.get('retrieve_ms', 0):.0f} ms")
                        col3.metric("Generate", f"{metrics.get('generate_ms', 0):.0f} ms")

                    # Save to history.
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "metrics": metrics,
                    })
                    st.session_state.last_eval_metrics = metrics

                except Exception as exc:
                    logger.error(f"Pipeline error for query '{query}': {exc}", exc_info=True)
                    error_msg = f"Error: {exc}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


def _run_rag_pipeline(
    query: str,
    settings: Settings,
    embedder: EmbeddingProvider,
    vector_store: PineconeVectorStore,
    generator: Generator,
    top_k: int,
) -> dict:
    """Execute the full RAG pipeline with timing and evaluation.

    Returns a metrics dict containing the answer, scores, and timings.
    """
    tracker = LatencyTracker()
    logger.info(f"Starting RAG pipeline for query: '{query}'")

    # Step 1: Retrieve (embed + search + rerank).
    with tracker.track("embed"):
        query_vector = embedder.embed_query(query)

    with tracker.track("retrieve"):
        results = vector_store.query(vector=query_vector, top_k=top_k)

    # Re-rank (included in retrieve timing for simplicity).
    from src.retriever import rerank
    results = rerank(query, results)

    # Step 2: Generate.
    with tracker.track("generate"):
        gen_result = generator.generate(query, results)

    answer = gen_result["answer"]
    timings = tracker.get_breakdown()

    # Step 3: Evaluate.
    relevance = score_retrieval_relevance(results)

    # Faithfulness scoring (separate LLM call — not timed as part of generation).
    faithfulness = score_faithfulness(query, results, answer, settings)

    # Gather active document sources.
    try:
        active_sources = vector_store.list_sources()
    except Exception:
        active_sources = []

    # Build metrics dict.
    metrics = {
        "query": query,
        "answer": answer,
        "retrieval_relevance": relevance,
        "faithfulness_score": faithfulness["score"],
        "faithfulness_justification": faithfulness["justification"],
        "embed_ms": timings.get("embed_ms", 0),
        "retrieve_ms": timings.get("retrieve_ms", 0),
        "generate_ms": timings.get("generate_ms", 0),
        "total_ms": timings.get("total_ms", 0),
        "num_documents": len(active_sources),
        "document_sources": active_sources,
        "top_k": top_k,
        "model_used": gen_result.get("model", ""),
    }

    # Persist to SQLite.
    log_evaluation(metrics, db_path=settings.eval_db_path)

    logger.info(f"RAG pipeline completed in {timings.get('total_ms', 0):.0f}ms")
    return metrics


# -- Tab 2: Documents ---------------------------------------------------------

def _render_documents_tab(
    settings: Settings,
    embedder: EmbeddingProvider,
    vector_store: PineconeVectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Render the document upload and management panel."""
    
    col_upload, col_docs = st.columns([1, 1], gap="large")

    # -- Upload Panel ---------------------------------------------------------
    with col_upload:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="saas-card-header"><span class="material-symbols-rounded">cloud_upload</span> Upload Documents</div>', unsafe_allow_html=True)
        st.caption("Supported formats: PDF, TXT, DOCX")

        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="doc_uploader",
            label_visibility="collapsed",
        )

        if uploaded_files and st.button("Process and Index", type="primary", use_container_width=True):
            for uploaded_file in uploaded_files:
                logger.info(f"Processing uploaded document: {uploaded_file.name}")
                with st.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
                    try:
                        # Step 1: Load.
                        st.write("Extracting text...")
                        pages = load_document(uploaded_file)
                        st.write(f"Extracted {len(pages)} page(s)")

                        # Step 2: Chunk.
                        st.write("Splitting into chunks...")
                        chunks = chunk_document(
                            pages,
                            chunk_size=chunk_size,
                            overlap=chunk_overlap,
                        )
                        st.write(f"Created {len(chunks)} chunk(s)")

                        # Step 3: Embed.
                        st.write("Generating embeddings...")
                        texts = [c["text"] for c in chunks]
                        embeddings = embedder.embed_batch(texts)
                        st.write(f"Generated {len(embeddings)} embedding(s)")

                        # Step 4: Upsert.
                        st.write("Uploading to vector store...")
                        count = vector_store.upsert_chunks(chunks, embeddings)
                        st.write(f"Indexed {count} vector(s)")

                        status.update(
                            label=f"{uploaded_file.name} — {count} chunks indexed",
                            state="complete",
                        )

                    except DocumentLoadError as exc:
                        logger.error(f"Failed to load document {uploaded_file.name}: {exc}")
                        status.update(label=f"Failed: {uploaded_file.name}", state="error")
                        st.error(str(exc))
                    except Exception as exc:
                        logger.error(f"Unexpected error processing {uploaded_file.name}: {exc}", exc_info=True)
                        status.update(label=f"Failed: {uploaded_file.name}", state="error")
                        st.error(f"Unexpected error: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)

    # -- Active Documents Panel -----------------------------------------------
    with col_docs:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="saas-card-header"><span class="material-symbols-rounded">folder_open</span> Active Documents</div>', unsafe_allow_html=True)
        try:
            sources = vector_store.list_sources()
        except Exception:
            sources = []
            st.warning("Could not fetch document list.")

        if not sources:
            st.markdown(
                '<div class="empty-state">'
                '<span class="material-symbols-rounded empty-icon">description</span>'
                '<div class="empty-title">No documents indexed</div>'
                '<div class="empty-desc">Upload files to build your knowledge base.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"{len(sources)} document(s) in knowledge base")
            for source in sources:
                with st.container(border=True):
                    col_name, col_btn = st.columns([5, 1])
                    with col_name:
                        st.markdown(
                            f'''<div style="display:flex; flex-direction:column; gap:4px;">
                                <div style="font-weight:600; font-size:15px; color:var(--text-primary); display:flex; align-items:center; gap:8px;">
                                    <span class="material-symbols-rounded" style="color:var(--accent); font-size:18px;">description</span>
                                    {source}
                                </div>
                                <div style="font-size:13px; color:var(--text-muted); display:flex; gap:12px; align-items:center;">
                                    <span>Indexed Document</span>
                                    <span class="status-badge">Ready</span>
                                </div>
                            </div>''',
                            unsafe_allow_html=True
                        )
                    with col_btn:
                        st.markdown('<div class="btn-delete">', unsafe_allow_html=True)
                        if st.button("Delete", key=f"del_{source}"):
                            with st.spinner(f"Removing..."):
                                verified = vector_store.delete_by_source(source)
                                if verified:
                                    st.success(f"Deleted")
                                else:
                                    st.warning("Refresh needed")
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # -- System Status Fill -----------------------------------------------
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="saas-card-header"><span class="material-symbols-rounded">memory</span> System Status</div>', unsafe_allow_html=True)
        st.markdown(
            f'''<div style="display:flex; flex-direction:column; gap:12px; font-size:14px; color:var(--text-secondary);">
                <div style="display:flex; justify-content:space-between;"><span>Vector Database:</span> <span style="color:var(--success); font-weight:600;">Connected</span></div>
                <div style="display:flex; justify-content:space-between;"><span>Index Name:</span> <span style="color:var(--text-primary);">{settings.active_pinecone_index}</span></div>
                <div style="display:flex; justify-content:space-between;"><span>Embedding Model:</span> <span style="color:var(--text-primary);">{settings.embedding_provider}</span></div>
                <div style="display:flex; justify-content:space-between;"><span>LLM Generator:</span> <span style="color:var(--text-primary);">{settings.llm_provider}</span></div>
            </div>''',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


# -- Tab 3: Evaluation Dashboard ----------------------------------------------

def _render_evaluation_tab(settings: Settings) -> None:
    """Render the evaluation metrics dashboard."""
    st.header("Evaluation Dashboard")

    records = get_recent_evaluations(limit=50, db_path=settings.eval_db_path)

    if not records:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-title">No evaluation data yet</div>'
            '<div class="empty-desc">Ask questions in the Chat tab to start collecting metrics.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Reverse so oldest is first (for chronological charts).
    records = list(reversed(records))

    # -- Summary Statistics ---------------------------------------------------
    st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    relevance_scores = [r["retrieval_relevance"] for r in records if r["retrieval_relevance"] is not None]
    faithfulness_scores = [r["faithfulness_score"] for r in records if r["faithfulness_score"] is not None]
    total_latencies = [r["total_ms"] for r in records if r["total_ms"] is not None]

    col1.metric("Queries Logged", len(records))
    col2.metric(
        "Avg Relevance",
        f"{sum(relevance_scores) / len(relevance_scores):.3f}" if relevance_scores else "N/A",
    )
    col3.metric(
        "Avg Faithfulness",
        f"{sum(faithfulness_scores) / len(faithfulness_scores):.1f}/5" if faithfulness_scores else "N/A",
    )
    col4.metric(
        "Avg Latency",
        f"{sum(total_latencies) / len(total_latencies):.0f} ms" if total_latencies else "N/A",
    )

    st.markdown("---")

    # -- Charts ---------------------------------------------------------------
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="chart-header">Retrieval Relevance Over Time</div>', unsafe_allow_html=True)
        if relevance_scores:
            st.line_chart(
                data={"Relevance": relevance_scores},
                use_container_width=True,
            )

        st.markdown('<div class="chart-header">Faithfulness Scores Over Time</div>', unsafe_allow_html=True)
        if faithfulness_scores:
            st.line_chart(
                data={"Faithfulness": faithfulness_scores},
                use_container_width=True,
            )

    with chart_col2:
        st.markdown('<div class="chart-header">Latency Breakdown (ms)</div>', unsafe_allow_html=True)
        latency_data = []
        for r in records:
            if r["total_ms"] is not None:
                latency_data.append({
                    "Embed": r.get("embed_ms", 0) or 0,
                    "Retrieve": r.get("retrieve_ms", 0) or 0,
                    "Generate": r.get("generate_ms", 0) or 0,
                })
        if latency_data:
            st.bar_chart(
                data=latency_data,
                use_container_width=True,
            )

    st.markdown("---")

    # -- Before/After Comparison ----------------------------------------------
    st.markdown('<div class="section-label">Before / After Comparison</div>', unsafe_allow_html=True)
    st.caption("Select two query records to compare their metrics.")

    if len(records) >= 2:
        query_labels = [
            f"#{r.get('id', i)} — {r['query'][:60]}..."
            for i, r in enumerate(records)
        ]

        col_before, col_after = st.columns(2)
        with col_before:
            before_idx = st.selectbox(
                "Before (older)", range(len(records)),
                format_func=lambda i: query_labels[i],
            )
        with col_after:
            after_idx = st.selectbox(
                "After (newer)", range(len(records)),
                index=min(1, len(records) - 1),
                format_func=lambda i: query_labels[i],
            )

        if st.button("Compare", type="primary"):
            comparison = compare_states(records[before_idx], records[after_idx])

            for key, vals in comparison.items():
                label = key.replace("_", " ").title()
                delta = vals["delta"]
                delta_str = f"{delta:+.4f}"
                css_class = "delta-positive" if vals["improved"] else "delta-negative"
                arrow = "^" if vals["improved"] else "v"
                st.markdown(
                    f'**{label}**: {vals["before"]:.4f} &rarr; {vals["after"]:.4f} '
                    f'<span class="{css_class}">({delta_str})</span>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Need at least 2 query records for comparison.")

    st.markdown("---")

    # -- Recent Query Log -----------------------------------------------------
    st.markdown('<div class="section-label">Recent Query Log</div>', unsafe_allow_html=True)
    for r in reversed(records[-10:]):  # Most recent 10
        with st.expander(f"Q: {r['query'][:80]}..."):
            st.markdown(f"**Answer:** {r['answer'][:300]}...")

            # Metric badges row
            rel = r.get("retrieval_relevance", 0) or 0
            faith = r.get("faithfulness_score", 0) or 0
            lat = r.get("total_ms", 0) or 0
            badges = _relevance_badge(rel) + _faithfulness_badge(faith) + _latency_badge(lat)
            st.markdown(badges, unsafe_allow_html=True)

            st.markdown(f"**Justification:** {r.get('faithfulness_justification', 'N/A')}")
            st.markdown(
                f"**Latency:** {lat:.0f}ms total "
                f"(embed: {r.get('embed_ms', 0) or 0:.0f}ms, "
                f"retrieve: {r.get('retrieve_ms', 0) or 0:.0f}ms, "
                f"generate: {r.get('generate_ms', 0) or 0:.0f}ms)"
            )
            st.markdown(f"**Model:** {r.get('model_used', 'N/A')}")
            sources_raw = r.get("document_sources", "[]")
            try:
                sources_list = json.loads(sources_raw) if isinstance(sources_raw, str) else sources_raw
            except json.JSONDecodeError:
                sources_list = []
            st.markdown(f"**Active Docs:** {', '.join(sources_list) if sources_list else 'N/A'}")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    """Application entry point — initialise resources and render tabs."""
    logger.info("Starting Streamlit application...")
    _init_session_state()

    # Load shared resources (cached — only runs once).
    try:
        settings = init_settings()
    except ValueError as exc:
        logger.error(f"Configuration error during startup: {exc}")
        st.error(f"Configuration Error: {exc}")
        st.info("Please check your `.env` file and ensure all required API keys are set.")
        st.stop()

    embedder = init_embedder(settings)
    vector_store = init_vector_store(settings)
    generator = init_generator(settings)

    # Sidebar.
    tunables = _render_sidebar(settings, vector_store)

    # Tabs.
    tab_chat, tab_docs, tab_eval = st.tabs(["Chat", "Documents", "Evaluation"])

    with tab_chat:
        _render_chat_tab(
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            generator=generator,
            top_k=tunables["top_k"],
        )

    with tab_docs:
        _render_documents_tab(
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            chunk_size=tunables["chunk_size"],
            chunk_overlap=tunables["chunk_overlap"],
        )

    with tab_eval:
        _render_evaluation_tab(settings)


if __name__ == "__main__":
    main()
