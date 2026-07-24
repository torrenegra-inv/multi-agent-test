"""
grounding_tool.py
-----------------
Local drop-in replacement for SAP's RetrievalAPIClient grounding service.

Uses sentence-transformers for embeddings and cosine similarity for vector
search — the same RAG concept as SAP's grounding pipeline, running entirely
locally with no external API calls.

Requirements:
    pip install sentence-transformers
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from crewai.tools import tool
except ImportError:
    def tool(name):
        def decorator(fn): return fn
        return decorator

# ---------------------------------------------------------------------------
# Document store — loaded once at import time
# ---------------------------------------------------------------------------

# Evidence documents folder: same directory as this file
DOCS_DIR = Path(__file__).parent / "evidence_documents"

_documents: list[dict] = []       # [{source, text, embedding}, ...]
_embedder = None                   # lazy-loaded SentenceTransformer


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("[grounding] Loading embedding model (first run only)...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[grounding] Embedding model ready.")
    return _embedder


def _load_documents():
    """Read all .txt files from the evidence_documents folder and embed them."""
    global _documents
    if _documents:
        return  # already loaded

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Evidence documents folder not found: {DOCS_DIR}\n"
            "Make sure 'evidence_documents/' is in the same folder as this script."
        )

    embedder = _get_embedder()
    txt_files = sorted(DOCS_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {DOCS_DIR}")

    print(f"[grounding] Indexing {len(txt_files)} evidence documents...")
    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        # Split into ~500-word chunks (mirrors how SAP's pipeline chunks documents)
        chunks = _chunk_text(text, max_words=500)
        for i, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk, normalize_embeddings=True)
            _documents.append({
                "source": path.name,
                "chunk_index": i,
                "text": chunk,
                "embedding": embedding,
            })

    print(f"[grounding] Indexed {len(_documents)} chunks from {len(txt_files)} documents.")


def _chunk_text(text: str, max_words: int = 500) -> list[str]:
    """Split text into overlapping chunks of ~max_words words."""
    words = text.split()
    chunks = []
    step = max_words - 50  # 50-word overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks if chunks else [text]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # embeddings are already normalized


def _search(query: str, top_k: int = 5) -> list[dict]:
    """Return the top-k most relevant document chunks for the query."""
    _load_documents()
    embedder = _get_embedder()
    query_vec = embedder.encode(query, normalize_embeddings=True)

    scored = [
        {
            "source": doc["source"],
            "chunk_index": doc["chunk_index"],
            "text": doc["text"],
            "score": _cosine_similarity(query_vec, doc["embedding"]),
        }
        for doc in _documents
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# CrewAI tool
# ---------------------------------------------------------------------------

@tool("call_grounding_service")
def call_grounding_service(user_question: str) -> str:
    """Search the evidence document database and retrieve relevant information.

    This tool performs semantic (vector) search over all available evidence
    documents, returning the most relevant passages to ground the agent's
    analysis in real facts rather than hallucinated information.

    Use this tool to look up information about suspects (Sophie Dubois,
    Marcus Chen, Viktor Petrov), security logs, bank records, phone records,
    or any other evidence from the museum theft investigation.

    Args:
        user_question: A natural language question or search query, e.g.
                       "What evidence exists about Marcus Chen?" or
                       "What do the bank records show about Sophie Dubois?"

    Returns:
        JSON string with the top 5 most relevant document excerpts,
        including source filename and relevance score.
    """
    try:
        results = _search(user_question, top_k=5)
        response = {
            "query": user_question,
            "results": [
                {
                    "source": r["source"],
                    "score": round(r["score"], 4),
                    "text": r["text"],
                }
                for r in results
            ]
        }
        return json.dumps(response, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    queries = [
        "What evidence exists about Marcus Chen?",
        "What do the bank records show about Sophie Dubois?",
        "What is Viktor Petrov's criminal history?",
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        results = _search(q, top_k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['source']}: {r['text'][:200]}...")
