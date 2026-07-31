# 🧠 Dynamic RAG Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45%2B-FF4B4B?logo=streamlit&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-5.0%2B-000000?logo=pinecone&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Supported-F55036?logo=groq&logoColor=white)

A production-quality **Retrieval-Augmented Generation (RAG)** system built as a modern Streamlit application. Upload your documents (PDF, TXT, DOCX), ask questions in a seamless chat interface, and receive highly accurate answers grounded entirely in your documents. 

The application goes beyond simple question answering by providing a **live evaluation dashboard** that tracks retrieval quality, system latency, and answer faithfulness for every single query!

---

## ✨ Key Features

- **Live Knowledge Base:** Add or delete documents on the fly; the Pinecone vector index stays perfectly in sync.
- **Structure-Aware Chunking:** Custom three-tier splitting strategy (paragraph → sentence → character) to preserve semantic coherence without losing context.
- **Hybrid Re-Ranking:** Blends cosine similarity (semantic) with exact keyword overlap (lexical) for pinpoint precision.
- **Swappable Providers:** Seamlessly toggle between HuggingFace/OpenAI for embeddings and Groq/OpenAI for LLM inference via simple config changes.
- **Built-in Evaluation:** Deep insights for every query including retrieval relevance, latency breakdown, and faithfulness scoring (LLM-as-a-Judge).
- **Historical Dashboard:** A dedicated SQLite-backed trends page to track before/after comparisons as your knowledge base grows.

---

## 🛠️ Technology Stack

- **Frontend UI:** `streamlit`
- **Vector Database:** `pinecone`
- **LLM Inference:** `groq`, `openai`
- **Embeddings:** `sentence-transformers` (powered by `torch` and `numpy`), `openai`
- **Document Processing:** `langchain-community`, `pypdf`, `docx2txt`
- **Storage:** Local `sqlite3` for evaluation logging

---

## 🏗️ Architecture

```text
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
```

### File Structure Breakdown

```text
├── .env.example          # Environment variable template
├── requirements.txt      # Pinned project dependencies
├── config.py             # Centralized settings (reads .env)
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

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- A [Pinecone](https://www.pinecone.io/) account (Free tier works perfectly — requires 2 indexes)
- A [Groq](https://console.groq.com/) API key
- *(Optional)* An [OpenAI](https://platform.openai.com/) API key if you want to use OpenAI models

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/moazhassan751/dynamic-rag-knowledge-assistant.git
cd dynamic-rag-knowledge-assistant

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Configure your environment variables
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# 5. Open .env in a text editor and add your API keys:
# PINECONE_API_KEY=your_pinecone_key
# GROQ_API_KEY=your_groq_key
```

### Running the Application

```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## ⚙️ Configuration (.env)

All core settings are controlled via the `.env` file, allowing you to seamlessly swap out infrastructure without changing code:

| Variable | Default | Description |
|----------|---------|-------------|
| `PINECONE_API_KEY` | *(required)* | Pinecone API key |
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `OPENAI_API_KEY` | *(optional)* | OpenAI API key (for swappable providers) |
| `EMBEDDING_PROVIDER` | `huggingface` | Options: `huggingface` (384-dim) or `openai` (1536-dim) |
| `LLM_PROVIDER` | `groq` | Options: `groq` or `openai` |
| `PINECONE_INDEX_384` | `rag-384` | Index name for HuggingFace embeddings |
| `PINECONE_INDEX_1536` | `rag-1536` | Index name for OpenAI embeddings |

---

## 📐 Design Decisions

### 1. The "LangChain Boundary"
LangChain is an incredibly powerful tool, but heavily abstracting logic can make it difficult to debug. In this project, LangChain is used **only** in `document_loader.py` for parsing PDFs, TXTs, and DOCXs. Every other component (chunking, embeddings, vector storage, retrieval, generation, and evaluation) is implemented completely from scratch. 

### 2. Dual Pinecone Indexes
Because Pinecone fixes the vector dimension at the time of index creation, we cannot store 384-dimensional HuggingFace vectors and 1536-dimensional OpenAI vectors in the same index. We maintain two separate indexes to allow seamless switching between embedding providers in `.env`.

### 3. Hybrid Re-Ranking
After initial vector search, the results are re-ranked using a hybrid blend of cosine similarity (70%) and keyword-overlap Jaccard similarity (30%). This strategy captures both the deep semantic relevance of vector embeddings and exact lexical terminology matches.

### 4. Faithfulness Evaluation (LLM-as-a-Judge)
A secondary LLM call acts as an impartial "judge", scoring whether the generated answer is genuinely supported by the retrieved context on a 1-5 scale. This automated evaluation catches AI hallucinations in real-time without requiring tedious human monitoring.

---

## 📄 License
This project was built as a  internship deliverable. Feel free to use and adapt the architecture for your own RAG solutions!
