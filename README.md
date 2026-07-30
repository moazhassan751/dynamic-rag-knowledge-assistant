# 🧠 Dynamic RAG Knowledge Assistant

A production-quality **Retrieval-Augmented Generation (RAG)** system built as a Streamlit application. Upload documents (PDF/TXT/DOCX), ask questions in a chat interface, and receive answers grounded in your documents — with a live evaluation dashboard tracking retrieval quality, latency, and answer faithfulness.

## ✨ Key Features

- **Live Knowledge Base** — Add or delete documents at any time; the vector index stays in sync
- **Structure-Aware Chunking** — Three-tier splitting (paragraph → sentence → character) preserves semantic coherence
- **Hybrid Re-Ranking** — Blends cosine similarity with keyword overlap for precision
- **Swappable Providers** — Toggle between HuggingFace/OpenAI embeddings and Groq/OpenAI LLMs via config
- **Built-in Evaluation** — Per-query metrics: retrieval relevance, latency breakdown, faithfulness scoring
- **Historical Dashboard** — SQLite-backed trends, before/after comparisons across knowledge-base states

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Streamlit  │────▶│   Document   │────▶│   Chunker    │
│   (app.py)   │     │   Loader     │     │  (from       │
│              │     │  (LangChain  │     │   scratch)   │
│  Chat │ Docs │     │   ONLY)      │     └──────┬───────┘
│  Eval │      │     └──────────────┘            │
└──┬────┴──────┘                           ┌─────▼───────┐
   │                                       │  Embedder   │
   │  ┌──────────────┐                    │  (HF/OpenAI │
   ├─▶│  Retriever   │◀───────────────────│   direct)   │
   │  │  + Re-ranker │                    └─────┬───────┘
   │  └──────┬───────┘                          │
   │         │                             ┌────▼────────┐
   │   ┌─────▼───────┐                    │  Pinecone   │
   ├──▶│  Generator  │                    │  (raw SDK)  │
   │   │  (Groq/     │                    └─────────────┘
   │   │   OpenAI)   │
   │   └─────────────┘
   │
   ├──▶ Evaluator ──▶ SQLite (eval_logs.db)
   │
```

### File Structure

```
├── .env.example          # Environment variable template
├── .gitignore
├── requirements.txt      # Pinned dependencies
├── config.py             # Centralised settings (reads .env)
├── app.py                # Streamlit UI (entry point)
│
├── src/
│   ├── document_loader.py  # LangChain loaders ONLY
│   ├── chunking.py         # Structure-aware chunking (from scratch)
│   ├── embeddings.py       # HuggingFace/OpenAI direct SDK (from scratch)
│   ├── vector_store.py     # Raw Pinecone client (from scratch)
│   ├── retriever.py        # Query + re-ranking (from scratch)
│   ├── generator.py        # Prompt + Groq/OpenAI direct (from scratch)
│   └── evaluator.py        # Metrics + SQLite persistence (from scratch)
```

## 🚀 Setup

### Prerequisites
- Python 3.11+
- A [Pinecone](https://www.pinecone.io/) account (free tier works — need 2 indexes)
- A [Groq](https://console.groq.com/) API key
- (Optional) An [OpenAI](https://platform.openai.com/) API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/moazhassan751/dynamic-rag-knowledge-assistant.git
cd dynamic-rag-knowledge-assistant

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# 5. Edit .env with your API keys
# PINECONE_API_KEY=your-key
# GROQ_API_KEY=your-key
```

### Running

```bash
streamlit run app.py
```

## ⚙️ Configuration

All settings are controlled via the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `PINECONE_API_KEY` | *(required)* | Pinecone API key |
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `OPENAI_API_KEY` | *(optional)* | OpenAI API key (for swappable providers) |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` (384-dim) or `openai` (1536-dim) |
| `LLM_PROVIDER` | `groq` | `groq` or `openai` |
| `PINECONE_INDEX_384` | `rag-384` | Index name for HuggingFace embeddings |
| `PINECONE_INDEX_1536` | `rag-1536` | Index name for OpenAI embeddings |

## 📐 Design Decisions

### LangChain Boundary
LangChain is used **only** in `document_loader.py` for PDF/TXT/DOCX parsing. Every other component (chunking, embeddings, vector store, retrieval, generation, evaluation) is implemented from scratch to demonstrate understanding of each layer.

### Dual Pinecone Indexes
Pinecone fixes vector dimension at index creation time. Since HuggingFace (384-dim) and OpenAI (1536-dim) produce different-sized vectors, we maintain two separate indexes and select the correct one based on the active embedding provider.

### Hybrid Re-Ranking
After vector search, results are re-ranked using a blend of cosine similarity (70%) and keyword-overlap Jaccard similarity (30%). This captures both semantic relevance and exact terminology matches.

### Faithfulness Evaluation
A second LLM call (same provider) acts as a judge, scoring whether the generated answer is supported by the retrieved context on a 1-5 scale. This lightweight "LLM-as-judge" approach catches hallucinations without requiring human evaluation.

## 📄 License

This project was built as a university internship deliverable.
