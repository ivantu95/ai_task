"""
Semantic search layer - the same RAG building blocks as the Netflix RAG
project (HuggingFace sentence-transformer embeddings + FAISS), applied
here to search over a personal task list instead of a movie catalog.

Since a personal user has at most a few hundred tasks, we build a fresh
in-memory FAISS index per search instead of persisting one to disk. This
avoids sync complexity (no need to update an index every time a task is
added/completed) and is fast enough at this scale.
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def semantic_search(
    query: str, candidates: list[dict], text_key: str = "text", top_k: int = 5, threshold: float = 0.3
) -> list[dict]:
    """Returns candidates most semantically similar to the query, above a
    minimum similarity threshold, sorted best match first."""
    if not candidates:
        return []

    model = _get_model()
    texts = [c[text_key] for c in candidates]
    embeddings = model.encode(texts, normalize_embeddings=True)
    query_embedding = model.encode([query], normalize_embeddings=True)

    # Inner product on normalized vectors is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))

    k = min(top_k, len(candidates))
    scores, indices = index.search(np.array(query_embedding, dtype="float32"), k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        results.append(candidates[idx])
    return results