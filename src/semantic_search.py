"""
Модуль поиска по задачам и истории.

Поддерживает 2 режима:
1. Легковесный режим (по умолчанию, USE_LOCAL_EMBEDDINGS=false):
   Использует быстрый алгоритм ранжирования по совпадению ключевых слов и фраз.
   Не загружает PyTorch, FAISS и веса нейросетей, потребляя 0 МБ дополнительной памяти
   и 0% CPU. Идеально подходит для работы на простых серверах и VPS.
2. Локальный нейросетевой режим (USE_LOCAL_EMBEDDINGS=true):
   Оригинальная архитектура на SentenceTransformers (all-MiniLM-L6-v2) и in-memory FAISS.
   Используется для локального вычисления векторных эмбеддингов на более мощных машинах.
"""
import os

USE_LOCAL_EMBEDDINGS = os.environ.get("USE_LOCAL_EMBEDDINGS", "false").lower() in ("true", "1", "yes")

_model = None


def _get_model():
    """Загружает модель SentenceTransformer только при явном включении локального режима."""
    global _model
    if _model is None:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass

        from sentence_transformers import SentenceTransformer
        print("[Search] Загрузка локальной модели эмбеддингов (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _local_faiss_semantic_search(
    query: str, candidates: list[dict], text_key: str = "text", top_k: int = 5, threshold: float = 0.3
) -> list[dict]:
    """Оригинальный локальный семантический поиск через SentenceTransformers и FAISS."""
    try:
        import numpy as np
        import faiss

        model = _get_model()
        texts = [c.get(text_key, "") for c in candidates]
        embeddings = model.encode(texts, normalize_embeddings=True)
        query_embedding = model.encode([query], normalize_embeddings=True)

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

    except Exception as e:
        print(f"[Search] Ошибка локального FAISS поиска ({e}), откат на легковесный...")
        return _lightweight_search(query, candidates, text_key, top_k)


def _lightweight_search(
    query: str, candidates: list[dict], text_key: str = "text", top_k: int = 5
) -> list[dict]:
    """Быстрый поиск без загрузки PyTorch и эмбеддингов.
    Ранжирует результаты по точному совпадению и совпадению ключевых слов."""
    if not candidates or not query:
        return []

    q_lower = query.strip().lower()
    query_words = [w for w in q_lower.split() if len(w) > 1]
    if not query_words:
        query_words = [q_lower]

    scored = []
    for c in candidates:
        text = str(c.get(text_key, "")).lower()
        if not text:
            continue

        score = 0.0
        # Точное вхождение всей поисковой фразы — максимальный приоритет
        if q_lower in text:
            score += 10.0

        # Вхождение отдельных слов из запроса
        matched_words = sum(1 for w in query_words if w in text)
        if matched_words > 0:
            score += matched_words * 2.0

        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def semantic_search(
    query: str, candidates: list[dict], text_key: str = "text", top_k: int = 5, threshold: float = 0.3
) -> list[dict]:
    """Основная точка входа для поиска по кандидатам.
    По умолчанию использует легковесный режим без нагрузки на сервер."""
    if not candidates or not query:
        return []

    if USE_LOCAL_EMBEDDINGS:
        return _local_faiss_semantic_search(query, candidates, text_key, top_k, threshold)

    return _lightweight_search(query, candidates, text_key, top_k)
