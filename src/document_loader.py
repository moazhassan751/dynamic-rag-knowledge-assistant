"""
Document Loader — extract raw text from PDF, TXT, and DOCX files.

THIS IS THE ONLY MODULE THAT USES LANGCHAIN.

LangChain's document loaders handle the messy details of binary format
parsing (PDF page extraction, DOCX XML unwrapping, encoding detection for
plain text). Reimplementing these parsers from scratch would add complexity
without demonstrating RAG understanding — that's why the project constraint
allows LangChain here but nowhere else.

Public API
----------
load_document(uploaded_file) -> list[dict]
    Accepts a Streamlit UploadedFile, writes it to a temp file, parses it
    with the appropriate LangChain loader, and returns a list of page dicts.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from src.logger import setup_logger

logger = setup_logger(__name__)

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile


# ── Custom Exception ─────────────────────────────────────────────────────────

class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed.

    Wraps lower-level exceptions from LangChain loaders so that callers
    only need to catch one type.
    """


# ── Supported file types → loader mapping ────────────────────────────────────

_LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}

SUPPORTED_EXTENSIONS: list[str] = list(_LOADER_MAP.keys())


# ── Public API ───────────────────────────────────────────────────────────────

def load_document(uploaded_file: "UploadedFile") -> list[dict]:
    """Extract text and metadata from an uploaded document.

    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile
        The file object from Streamlit's file_uploader widget.

    Returns
    -------
    list[dict]
        Each dict represents one logical page/section::

            {
                "text": str,           # raw text content
                "metadata": {
                    "source": str,     # original filename
                    "page": int | None # 0-indexed page number (PDF only)
                }
            }

    Raises
    ------
    DocumentLoadError
        If the file type is unsupported or the loader fails.
    """
    filename: str = uploaded_file.name
    extension: str = Path(filename).suffix.lower()

    if extension not in _LOADER_MAP:
        logger.error(f"Unsupported file type '{extension}' for file '{filename}'.")
        raise DocumentLoadError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Write the in-memory upload to a temporary file so LangChain loaders
    # (which expect a filesystem path) can read it.
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
            prefix="rag_upload_",
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        loader_cls = _LOADER_MAP[extension]
        loader = loader_cls(tmp_path)
        logger.info(f"Loading '{filename}' via {loader_cls.__name__}")
        langchain_docs = loader.load()
        logger.info(f"Loaded {len(langchain_docs)} raw pages from '{filename}'")

    except DocumentLoadError:
        raise  # Re-raise our own errors as-is
    except Exception as exc:
        logger.error(f"Failed to load '{filename}': {exc}", exc_info=True)
        raise DocumentLoadError(
            f"Failed to load '{filename}': {exc}"
        ) from exc
    finally:
        # Always clean up the temp file, even if loading fails.
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── Convert LangChain Document objects → plain dicts ─────────────────
    # This is the boundary where LangChain ends and our own code begins.
    # Downstream modules never see LangChain types.
    pages: list[dict] = []
    for i, doc in enumerate(langchain_docs):
        text = doc.page_content.strip()
        if not text:
            continue  # Skip empty pages (common in scanned PDFs)

        page_number: int | None = None
        if "page" in doc.metadata:
            page_number = int(doc.metadata["page"])

        pages.append({
            "text": text,
            "metadata": {
                "source": filename,
                "page": page_number,
            },
        })

    if not pages:
        logger.warning(f"No text content extracted from '{filename}'.")
        raise DocumentLoadError(
            f"No text content could be extracted from '{filename}'. "
            "The file may be empty or contain only images."
        )

    return pages
