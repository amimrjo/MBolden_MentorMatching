"""
Embedding layer. Swappable: this is the one place that touches the model,
so upgrading to a bigger model or a hosted embeddings API later is a one-file change.

FALLBACK NOTE: this sandbox's network allowlist doesn't include huggingface.co,
so the real sentence-transformer weights can't be downloaded here. get_model()
tries the real model first and falls back to a deterministic hashed
bag-of-words encoder so the pipeline still runs end-to-end for the demo. In any
normal deployment (with HF access, or weights baked into the image/container)
USE_FALLBACK stays False and every downstream call is unaffected -- matching.py,
main.py, etc. never know which encoder produced the vectors.
"""
import hashlib
import re
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for CV/bio semantic matching
FALLBACK_DIM = 384  # matches all-MiniLM-L6-v2's output size, so nothing downstream cares

USE_FALLBACK = False


@lru_cache(maxsize=1)
def get_model():
    global USE_FALLBACK
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        print(f"[embeddings] couldn't load {MODEL_NAME} ({exc}); using local fallback encoder")
        USE_FALLBACK = True
        return None


def _fallback_encode(text: str) -> list[float]:
    """
    Deterministic hashed bag-of-words vector, L2-normalized. Not a real
    semantic embedding -- words are hashed independently so it can't capture
    meaning the way a transformer does -- but it preserves exact keyword
    overlap, which is enough to demo the retrieval + ranking + capacity logic
    end-to-end while offline. Swap out automatically once the real model loads.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    vec = np.zeros(FALLBACK_DIM, dtype=np.float32)
    for w in words:
        h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
        idx = h % FALLBACK_DIM
        sign = 1.0 if (h // FALLBACK_DIM) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def embed_text(text: str) -> list[float]:
    model = get_model()
    if model is None:
        return _fallback_encode(text)
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_model()
    if model is None:
        return [_fallback_encode(t) for t in texts]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    # vectors are already normalized, so dot product == cosine similarity
    return float(np.dot(a, b))
