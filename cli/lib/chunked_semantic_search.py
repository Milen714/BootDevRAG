from datetime import datetime, timezone
from typing import Any

import numpy as np

from .config import (
    CHUNKED_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    CHUNKING_STRATEGY_VERSION,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_INPUT_MODE,
    EMBEDDING_INPUT_TEMPLATE,
    EMBEDDING_INPUT_TEMPLATE_VERSION,
    EMBEDDING_MANIFEST_PATH,
    EMBEDDING_SAFETY_MARGIN_TOKENS,
    EMBEDDING_TITLE_MAX_TOKENS,
    MODEL_NAME,
    MODEL_MAX_SEQUENCE_TOKENS,
    TOKENIZER_NAME,
)
from .io_utils import ensure_directory, read_jsonl, records_hash, write_json, write_jsonl
from .search_utils import SearchResult, format_search_result, load_chunks
from .semantic_search import EmbeddingArray, SemanticSearch, normalize_vectors


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model: Any | None = None,
        input_mode: str = EMBEDDING_INPUT_MODE,
    ) -> None:
        super().__init__(model_name, model=model)
        self.input_mode = input_mode
        self.chunk_embeddings: EmbeddingArray | None = None
        self.chunks: list[dict[str, Any]] = []

    def _manifest_matches(
        self, manifest: dict[str, Any], chunks: list[dict[str, Any]], embeddings: EmbeddingArray
    ) -> bool:
        return (
            manifest.get("embedding_model") == self.model_name
            and manifest.get("tokenizer") == TOKENIZER_NAME
            and manifest.get("chunking_strategy") == CHUNKING_STRATEGY_VERSION
            and manifest.get("chunk_max_tokens") == CHUNK_MAX_TOKENS
            and manifest.get("chunk_overlap_tokens") == CHUNK_OVERLAP_TOKENS
            and manifest.get("input_mode") == self.input_mode
            and manifest.get("input_template_version") == EMBEDDING_INPUT_TEMPLATE_VERSION
            and manifest.get("title_max_tokens") == EMBEDDING_TITLE_MAX_TOKENS
            and manifest.get("model_max_tokens") == MODEL_MAX_SEQUENCE_TOKENS
            and manifest.get("safety_margin_tokens") == EMBEDDING_SAFETY_MARGIN_TOKENS
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
        tokenizer = self.model.tokenizer
        model_limit = min(
            int(getattr(self.model, "max_seq_length", MODEL_MAX_SEQUENCE_TOKENS)),
            MODEL_MAX_SEQUENCE_TOKENS,
        )
        texts = [
            format_embedding_input(chunk, self.input_mode, tokenizer)
            for chunk in self.chunks
        ]
        token_counts = [
            len(tokenizer.encode(text, add_special_tokens=True)) for text in texts
        ]
        safe_limit = model_limit - EMBEDDING_SAFETY_MARGIN_TOKENS
        oversized = [
            (self.chunks[index]["chunk_id"], count)
            for index, count in enumerate(token_counts)
            if count > safe_limit
        ]
        if oversized:
            chunk_id, count = oversized[0]
            raise ValueError(
                f"Embedding input {chunk_id} has {count} tokens; safe limit is {safe_limit}"
            )
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
                "tokenizer": TOKENIZER_NAME,
                "embedding_shape": list(self.chunk_embeddings.shape),
                "chunk_count": len(self.chunks),
                "chunking_strategy": CHUNKING_STRATEGY_VERSION,
                "chunk_max_tokens": CHUNK_MAX_TOKENS,
                "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
                "input_mode": self.input_mode,
                "input_template_version": EMBEDDING_INPUT_TEMPLATE_VERSION,
                "title_max_tokens": EMBEDDING_TITLE_MAX_TOKENS,
                "model_max_tokens": MODEL_MAX_SEQUENCE_TOKENS,
                "safety_margin_tokens": EMBEDDING_SAFETY_MARGIN_TOKENS,
                "maximum_input_tokens": max(token_counts, default=0),
                "inputs_over_safe_limit": len(oversized),
                "corpus_hash": records_hash(self.chunks),
                "embedding_input_hash": records_hash(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "input": text,
                    }
                    for chunk, text in zip(self.chunks, texts)
                ),
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
                "chunk_token_count",
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


def truncate_to_tokens(text: str, max_tokens: int, tokenizer: Any) -> str:
    encoded = tokenizer(
        text or "", add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = encoded["offset_mapping"]
    if len(offsets) <= max_tokens:
        return (text or "").strip()
    return (text or "")[: offsets[max_tokens - 1][1]].strip()


def format_embedding_input(
    chunk: dict[str, Any], input_mode: str, tokenizer: Any
) -> str:
    text = str(chunk.get("text") or "").strip()
    if input_mode == "text":
        return text
    if input_mode == "title_text":
        title = truncate_to_tokens(
            str(chunk.get("title") or ""), EMBEDDING_TITLE_MAX_TOKENS, tokenizer
        )
        return EMBEDDING_INPUT_TEMPLATE.format(title=title, text=text)
    raise ValueError(f"Unsupported embedding input mode: {input_mode}")
