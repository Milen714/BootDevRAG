
import json
import os
from typing import Any

import numpy as np

from .search_utils import SearchResult, CACHE_DIR, CHUNKED_EMBEDDINGS_PATH, DOCUMENT_PREVIEW_LENGTH, JSON_METADATA_PATH, MAX_CHUNK_SIZE, MODEL_NAME, format_search_result
from .semantic_search import ChunkMetadata, ChunkScore, EmbeddingArray, SemanticSearch, cosine_similarity, semantic_chunk_text

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings: EmbeddingArray | None = None
        self.chunk_metadata: list[ChunkMetadata] | None = None

    def _chunk_config(self) -> dict[str, Any]:
        return {"max_chunk_size": MAX_CHUNK_SIZE, "overlap": 1, "strategy": "sliding_window_skip_overlap_tail", "movie_idx": "document_index"}

    def build_chunk_embeddings(self, documents: list[dict[str, Any]]) -> EmbeddingArray:
        if not documents:
            raise ValueError("Document list cannot be empty.")

        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks: list[str] = []
        metadata_list: list[ChunkMetadata] = []

        for movie_idx, doc in enumerate(documents):
            if doc["description"]:
                description_chunks = semantic_chunk_text(doc["description"], max_chunk_size=MAX_CHUNK_SIZE, overlap=1, verbose=False)
                all_chunks.extend(description_chunks)
                for idx in range(len(description_chunks)):
                    metadata_list.append(
                        ChunkMetadata(movie_idx=movie_idx, chunk_idx=idx, total_chunks=len(description_chunks))
                    )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata_list

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(CHUNKED_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(JSON_METADATA_PATH, "w") as f:
            json.dump({"chunks": self.chunk_metadata, 
                       "total_chunks": len(all_chunks),
                       "chunk_config": self._chunk_config()
                       },
                        f, indent=2
                        )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict[str, Any]]) -> EmbeddingArray:
        try:
            self.documents = documents
            self.document_map = {doc["id"]: doc for doc in documents}
            if os.path.exists(CHUNKED_EMBEDDINGS_PATH) and os.path.exists(JSON_METADATA_PATH):
                self.chunk_embeddings = np.load(CHUNKED_EMBEDDINGS_PATH)
                with open(JSON_METADATA_PATH, "r") as f:
                    metadata = json.load(f)
                    if metadata.get("chunk_config") != self._chunk_config() or metadata.get("total_chunks") != len(self.chunk_embeddings):
                        print("Chunk embeddings cache is stale. Rebuilding chunk embeddings...")
                        return self.build_chunk_embeddings(documents)
                    self.chunk_metadata = metadata.get("chunks", [])
                    return self.chunk_embeddings
            print("Chunk embeddings not found in cache. Building chunk embeddings...")
            return self.build_chunk_embeddings(documents)
        except Exception as e:
            print(f"Error loading or creating chunk embeddings: {e}")
            return np.array([])

    def search_chunks(self, query: str, limit: int = 10) -> list[SearchResult]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores: list[ChunkScore] = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": similarity,
                }
            )

        movie_scores: dict[int, float] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        if self.documents is None:
            raise ValueError(
                "No documents loaded. Call load_or_create_chunk_embeddings first."
            )
        results: list[SearchResult] = []
        for movie_idx, score in sorted_movies[:limit]:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )

        return results

