from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model

def embed_texts(texts: list) -> np.ndarray:
    model = get_embedder()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
