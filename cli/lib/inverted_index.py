import json
import math
import pickle
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .config import (
    BM25_B,
    BM25_CACHE_DIR,
    BM25_DOCMAP_PATH,
    BM25_DOC_LENGTHS_PATH,
    BM25_INDEX_PATH,
    BM25_K1,
    BM25_MANIFEST_PATH,
    BM25_TERM_FREQUENCIES_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
)
from .io_utils import ensure_directory, records_hash, write_json
from .search_utils import SearchResult, format_search_result, load_chunks, tokenize_text


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[str]] = {}
        self.docmap: dict[str, dict[str, Any]] = {}
        self.term_frequencies: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}

    @property
    def index_path(self) -> str:
        return str(BM25_INDEX_PATH)

    def _add_document(self, chunk_id: str, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies[chunk_id] = Counter(tokens)
        self.doc_lengths[chunk_id] = len(tokens)
        for token in set(tokens):
            self.index.setdefault(token, set()).add(chunk_id)

    def build(self, chunks: list[dict[str, Any]] | None = None) -> None:
        source_chunks = chunks if chunks is not None else load_chunks()
        self.index.clear()
        self.docmap.clear()
        self.term_frequencies.clear()
        self.doc_lengths.clear()
        for chunk in source_chunks:
            chunk_id = str(chunk["chunk_id"])
            self.docmap[chunk_id] = chunk
            searchable_text = " ".join(
                str(chunk.get(field) or "")
                for field in ("title", "theme", "keywords", "text")
            )
            self._add_document(chunk_id, searchable_text)

    def save(self) -> None:
        ensure_directory(BM25_CACHE_DIR)
        for path, value in (
            (BM25_INDEX_PATH, self.index),
            (BM25_DOCMAP_PATH, self.docmap),
            (BM25_TERM_FREQUENCIES_PATH, self.term_frequencies),
            (BM25_DOC_LENGTHS_PATH, self.doc_lengths),
        ):
            with path.open("wb") as file:
                pickle.dump(value, file)
        chunks = list(self.docmap.values())
        write_json(
            BM25_MANIFEST_PATH,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": len(chunks),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "corpus_hash": records_hash(chunks),
                "searchable_fields": ["title", "theme", "keywords", "text"],
            },
        )

    def cache_is_valid(self, chunks: list[dict[str, Any]]) -> bool:
        paths = (
            BM25_INDEX_PATH,
            BM25_DOCMAP_PATH,
            BM25_TERM_FREQUENCIES_PATH,
            BM25_DOC_LENGTHS_PATH,
            BM25_MANIFEST_PATH,
        )
        if not all(path.exists() for path in paths):
            return False
        try:
            manifest = json.loads(BM25_MANIFEST_PATH.read_text(encoding="utf-8"))
            return (
                manifest.get("chunk_count") == len(chunks)
                and manifest.get("chunk_size") == CHUNK_SIZE
                and manifest.get("chunk_overlap") == CHUNK_OVERLAP
                and manifest.get("corpus_hash") == records_hash(chunks)
            )
        except (OSError, ValueError):
            return False

    def load(self) -> None:
        for path, attribute in (
            (BM25_INDEX_PATH, "index"),
            (BM25_DOCMAP_PATH, "docmap"),
            (BM25_TERM_FREQUENCIES_PATH, "term_frequencies"),
            (BM25_DOC_LENGTHS_PATH, "doc_lengths"),
        ):
            if not path.exists():
                raise FileNotFoundError("BM25 index not found. Run the build command first.")
            with path.open("rb") as file:
                setattr(self, attribute, pickle.load(file))

    def load_or_build(self, chunks: list[dict[str, Any]] | None = None) -> None:
        source_chunks = chunks if chunks is not None else load_chunks()
        if self.cache_is_valid(source_chunks):
            self.load()
            return
        self.build(source_chunks)
        self.save()

    def get_documents(self, term: str) -> list[str]:
        return sorted(self.index.get(term, set()))

    def get_term_frequency(self, chunk_id: str, term: str) -> int:
        return self.term_frequencies.get(chunk_id, Counter())[term]

    def get_bm25_idf(self, term: str) -> float:
        document_count = len(self.docmap)
        frequency = len(self.index.get(term, set()))
        return math.log((document_count - frequency + 0.5) / (frequency + 0.5) + 1)

    def _average_document_length(self) -> float:
        return sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0.0

    def get_bm25_tf(
        self, chunk_id: str, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        frequency = self.get_term_frequency(chunk_id, term)
        if frequency == 0:
            return 0.0
        average_length = self._average_document_length()
        length_norm = 1 - b + b * (self.doc_lengths[chunk_id] / average_length)
        return (frequency * (k1 + 1)) / (frequency + k1 * length_norm)

    def bm25(self, chunk_id: str, term: str) -> float:
        return self.get_bm25_idf(term) * self.get_bm25_tf(chunk_id, term)

    def bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        query_tokens = tokenize_text(query)
        candidate_ids = set().union(*(self.index.get(token, set()) for token in query_tokens)) if query_tokens else set()
        scored = [
            (chunk_id, sum(self.bm25(chunk_id, token) for token in query_tokens))
            for chunk_id in candidate_ids
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        results: list[SearchResult] = []
        for chunk_id, score in scored[:limit]:
            if score <= 0:
                continue
            chunk = self.docmap[chunk_id]
            results.append(
                format_search_result(
                    doc_id=chunk_id,
                    title=str(chunk.get("title") or ""),
                    document=str(chunk.get("text") or ""),
                    score=score,
                    chunk_id=chunk_id,
                    parent_id=chunk["doc_id"],
                    **{
                        key: chunk.get(key)
                        for key in (
                            "chunk_index", "total_chunks", "chunk_word_count", "url",
                            "publisher", "date", "theme", "keywords", "citation",
                            "source_type", "local_path",
                        )
                    },
                )
            )
        return results
