from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MODEL_NAME
from .preprocessing import chunk_sentences, split_sentences


EmbeddingArray = NDArray[Any]


def normalize_vectors(vectors: EmbeddingArray) -> EmbeddingArray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        return array if norm == 0 else array / norm
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms != 0)


def cosine_similarity(vec1: EmbeddingArray, vec2: EmbeddingArray) -> float:
    first = normalize_vectors(vec1)
    second = normalize_vectors(vec2)
    if not np.any(first) or not np.any(second):
        return 0.0
    return float(first @ second)


class SemanticSearch:
    def __init__(self, model_name: str = MODEL_NAME, model: Any | None = None) -> None:
        self.model_name = model_name
        self.model = model or SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> EmbeddingArray:
        if not text.strip():
            raise ValueError("Input text cannot be empty or just whitespace.")
        return normalize_vectors(self.model.encode([text])[0])


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
    words = text.split()
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")
    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    verbose: bool = False,
) -> list[str]:
    del verbose
    return chunk_sentences(split_sentences(text), max_chunk_size, overlap)


def verify_model() -> None:
    model = SemanticSearch().model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(query: str) -> None:
    embedding = SemanticSearch().generate_embedding(query)
    print(f"Text: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    embed_text(query)


def verify_embeddings() -> None:
    from .chunked_semantic_search import ChunkedSemanticSearch

    search = ChunkedSemanticSearch()
    embeddings = search.load_or_create_chunk_embeddings()
    print(f"Embeddings shape: {embeddings.shape}")
