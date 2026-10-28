"""
Модуль семантического поиска по задачам и истории.
Использует SentenceTransformers и FAISS с отложенным (ленивым) импортом,
чтобы бот запускался мгновенно и не потреблял лишнюю память/процессор на сервере.
"""
_model = None


def _get_model():
    """Загружает модель SentenceTransformer только при реальной необходимости."""
    global _model
    if _model is None:
        try:
            import torch
            torch.set_num_threads(1)  # Ограничиваем 1 потоком для экономии CPU на сервере
        except ImportError:
            pass

        from sentence_transformers import SentenceTransformer
        print("[Search] Загрузка легковесной модели эмбеддингов (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def semantic_search(
    query: str, candidates: list[dict], text_key: str = "text", top_k: int = 5, threshold: float = 0.3
) -> list[dict]:
    """Возвращает задачи, наиболее близкие по смыслу к поисковому запросу.
    Если векторные библиотеки недоступны или перегружены, безопасно откатывается на текстовый поиск."""
    if not candidates or not query:
        return []

    try:
        import numpy as np
        import faiss

        model = _get_model()
        texts = [c.get(text_key, "") for c in candidates]
        embeddings = model.encode(texts, normalize_embeddings=True)
        query_embedding = model.encode([query], normalize_embeddings=True)

        # Скалярное произведение для нормализованных векторов эквивалентно косинусному сходству
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
        print(f"[Search] Предупреждение: семантический поиск переключен на текстовый: {e}")
        query_lower = query.lower()
        return [c for c in candidates if query_lower in c.get(text_key, "").lower()]
