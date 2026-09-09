import numpy as np

from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.hybrid_search import (
    limit_chunks_per_parent,
    rrf_combine_search_results,
)
from lib.inverted_index import InvertedIndex
from lib.search_utils import format_search_result


def chunk(chunk_id: str, parent_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": parent_id,
        "title": f"Source {parent_id}",
        "text": text,
        "chunk_index": 0,
        "total_chunks": 1,
        "chunk_word_count": len(text.split()),
        "url": f"https://example.test/{parent_id}",
        "publisher": "Publisher",
        "date": "2025",
        "theme": "AI policy",
        "keywords": "safety",
        "citation": "Citation",
        "source_type": "html",
        "local_path": "data/raw/source.html",
    }


def result(chunk_id: str, parent_id: str, score: float):
    return format_search_result(
        chunk_id,
        f"Source {parent_id}",
        f"Relevant text for {chunk_id}",
        score,
        chunk_id=chunk_id,
        parent_id=parent_id,
    )


def test_bm25_indexes_and_returns_chunks() -> None:
    chunks = [
        chunk("A::chunk-0000", "A", "deepfake protections for children"),
        chunk("B::chunk-0000", "B", "unrelated environmental policy"),
    ]
    index = InvertedIndex()
    index.build(chunks)

    results = index.bm25_search("deepfake children", limit=5)

    assert [item["id"] for item in results] == ["A::chunk-0000"]
    assert results[0]["document"] == chunks[0]["text"]
    assert results[0]["metadata"]["parent_id"] == "A"
    assert results[0]["metadata"]["url"] == chunks[0]["url"]


def test_rrf_fuses_matching_chunk_ids_and_preserves_single_rankings() -> None:
    bm25 = [result("A::chunk-0000", "A", 3.0), result("B::chunk-0000", "B", 2.0)]
    semantic = [result("B::chunk-0000", "B", 0.9), result("C::chunk-0000", "C", 0.8)]

    fused = rrf_combine_search_results(bm25, semantic, k=60)

    assert fused[0]["id"] == "B::chunk-0000"
    by_id = {item["id"]: item for item in fused}
    assert by_id["A::chunk-0000"]["metadata"]["semantic_rank"] is None
    assert by_id["C::chunk-0000"]["metadata"]["bm25_rank"] is None


def test_parent_cap_fills_results_from_other_documents() -> None:
    ranked = [
        result("A::chunk-0000", "A", 1.0),
        result("A::chunk-0001", "A", 0.9),
        result("A::chunk-0002", "A", 0.8),
        result("B::chunk-0000", "B", 0.7),
    ]

    selected = limit_chunks_per_parent(ranked, limit=4, max_per_parent=2)

    assert [item["id"] for item in selected] == [
        "A::chunk-0000",
        "A::chunk-0001",
        "B::chunk-0000",
    ]


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        del kwargs
        vectors = {
            "matching evidence": [1.0, 0.0],
            "other evidence": [0.0, 1.0],
            "query": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def test_semantic_search_returns_only_chunk_text_and_parent_metadata() -> None:
    chunks = [
        chunk("A::chunk-0000", "A", "matching evidence"),
        chunk("B::chunk-0000", "B", "other evidence"),
    ]
    search = ChunkedSemanticSearch(model_name="fake", model=FakeEmbeddingModel())
    search.chunks = chunks
    search.chunk_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    results = search.search_chunks("query", limit=1)

    assert results[0]["id"] == "A::chunk-0000"
    assert results[0]["document"] == "matching evidence"
    assert results[0]["metadata"]["parent_id"] == "A"
