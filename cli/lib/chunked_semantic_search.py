from datetime import datetime, timezone
from typing import Any

import numpy as np

from .config import (
    CHUNKED_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MANIFEST_PATH,
    MODEL_NAME,
)
from .io_utils import ensure_directory, read_jsonl, records_hash, write_json, write_jsonl
from .search_utils import SearchResult, format_search_result, load_chunks
from .semantic_search import EmbeddingArray, SemanticSearch, normalize_vectors


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = MODEL_NAME, model: Any | None = None) -> None:
        super().__init__(model_name, model=model)
        self.chunk_embeddings: EmbeddingArray | None = None
        self.chunks: list[dict[str, Any]] = []

    def _manifest_matches(
        self, manifest: dict[str, Any], chunks: list[dict[str, Any]], embeddings: EmbeddingArray
    ) -> bool:
        return (
            manifest.get("embedding_model") == self.model_name
            and manifest.get("chunk_size") == CHUNK_SIZE
            and manifest.get("chunk_overlap") == CHUNK_OVERLAP
            and manifest.get("chunk_count") == len(chunks) == len(embeddings)
            and manifest.get("corpus_hash") == records_hash(chunks)
            and manifest.get("embedding_shape") == list(embeddings.shape)
        )

    def build_chunk_embeddings(
        self, chunks: list[dict[str, Any]] | None = None
    ) -> EmbeddingArray:
        self.chunks = chunks if chunks is not None else load_chunks()
        if not self.chunks:
            raise ValueError("Chunk list cannot be empty.")
        texts = [str(chunk.get("text") or "") for chunk in self.chunks]
        embeddings = self.model.encode(
            texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=True
        )
        self.chunk_embeddings = normalize_vectors(embeddings).astype(np.float32)
        ensure_directory(CHUNKED_EMBEDDINGS_PATH.parent)
        np.save(CHUNKED_EMBEDDINGS_PATH, self.chunk_embeddings)
        write_jsonl(CHUNK_METADATA_PATH, self.chunks)
        write_json(
            EMBEDDING_MANIFEST_PATH,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding_model": self.model_name,
                "embedding_shape": list(self.chunk_embeddings.shape),
                "chunk_count": len(self.chunks),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "corpus_hash": records_hash(self.chunks),
                "normalized": True,
            },
        )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(
        self, chunks: list[dict[str, Any]] | None = None
    ) -> EmbeddingArray:
        source_chunks = chunks if chunks is not None else load_chunks()
        if all(
            path.exists()
            for path in (CHUNKED_EMBEDDINGS_PATH, CHUNK_METADATA_PATH, EMBEDDING_MANIFEST_PATH)
        ):
            try:
                embeddings = np.load(CHUNKED_EMBEDDINGS_PATH).astype(np.float32)
                cached_chunks = read_jsonl(CHUNK_METADATA_PATH)
                import json

                manifest = json.loads(EMBEDDING_MANIFEST_PATH.read_text(encoding="utf-8"))
                if cached_chunks == source_chunks and self._manifest_matches(
                    manifest, source_chunks, embeddings
                ):
                    self.chunks = source_chunks
                    self.chunk_embeddings = embeddings
                    return embeddings
            except (OSError, ValueError, KeyError):
                pass
        return self.build_chunk_embeddings(source_chunks)

    def search_chunks(self, query: str, limit: int = 10) -> list[SearchResult]:
        if self.chunk_embeddings is None or not self.chunks:
            raise ValueError("No chunk embeddings loaded.")
        if limit <= 0:
            return []
        query_embedding = self.generate_embedding(query)
        scores = self.chunk_embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:limit]
        results: list[SearchResult] = []
        for index in top_indices:
            chunk = self.chunks[int(index)]
            results.append(
                format_search_result(
                    doc_id=str(chunk["chunk_id"]),
                    title=str(chunk.get("title") or ""),
                    document=str(chunk.get("text") or ""),
                    score=float(scores[index]),
                    **chunk_metadata(chunk),
                )
            )
        return results


def chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "parent_id": chunk["doc_id"],
        **{
            key: chunk.get(key)
            for key in (
                "chunk_index",
                "total_chunks",
                "chunk_word_count",
                "url",
                "publisher",
                "date",
                "theme",
                "keywords",
                "citation",
                "source_type",
                "local_path",
            )
        },
    }
