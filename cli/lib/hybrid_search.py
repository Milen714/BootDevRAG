from typing import Literal, TypedDict

from .chunked_semantic_search import ChunkedSemanticSearch
from .config import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    K_VALUE,
    MAX_CHUNKS_PER_PARENT,
    MIN_RETRIEVAL_CANDIDATES,
    SEARCH_LIMIT_MULTIPLIER,
)
from .inverted_index import InvertedIndex
from .search_utils import SearchResult, format_search_result, load_chunks


class RRFSearchCommandResult(TypedDict):
    original_query: str
    enhanced_query: str | None
    enhance_method: Literal["spell", "expand", "rewrite"] | None
    query: str
    k: int
    rerank_method: Literal["individual", "batch", "cross_encoder"] | None
    reranked: bool
    results: list[SearchResult]


class WeightedSearchCommandResult(TypedDict):
    original_query: str
    query: str
    alpha: float
    results: list[SearchResult]


class HybridSearch:
    def __init__(self, chunks: list[dict] | None = None) -> None:
        self.chunks = chunks if chunks is not None else load_chunks()
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(self.chunks)
        self.idx = InvertedIndex()
        self.idx.load_or_build(self.chunks)

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        candidate_limit = max(MIN_RETRIEVAL_CANDIDATES, limit * SEARCH_LIMIT_MULTIPLIER)
        combined = combine_search_results(
            self._bm25_search(query, candidate_limit),
            self.semantic_search.search_chunks(query, candidate_limit),
            alpha,
        )
        return limit_chunks_per_parent(combined, limit)

    def rrf_search(
        self, query: str, k: int = K_VALUE, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        candidate_limit = max(MIN_RETRIEVAL_CANDIDATES, limit * SEARCH_LIMIT_MULTIPLIER)
        combined = rrf_combine_search_results(
            self._bm25_search(query, candidate_limit),
            self.semantic_search.search_chunks(query, candidate_limit),
            k,
        )
        return limit_chunks_per_parent(combined, limit)


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum, maximum = min(scores), max(scores)
    if maximum == minimum:
        return [1.0] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]


def combine_search_results(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    alpha: float = DEFAULT_ALPHA,
) -> list[SearchResult]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    bm25_scores = {
        result["id"]: score
        for result, score in zip(bm25_results, normalize_scores([r["score"] for r in bm25_results]))
    }
    semantic_scores = {
        result["id"]: score
        for result, score in zip(semantic_results, normalize_scores([r["score"] for r in semantic_results]))
    }
    source_results = {result["id"]: result for result in bm25_results + semantic_results}
    combined: list[SearchResult] = []
    for chunk_id in bm25_scores.keys() | semantic_scores.keys():
        source = source_results[chunk_id]
        bm25_score = bm25_scores.get(chunk_id, 0.0)
        semantic_score = semantic_scores.get(chunk_id, 0.0)
        combined.append(
            format_search_result(
                doc_id=chunk_id,
                title=source["title"],
                document=source["document"],
                score=hybrid_score(bm25_score, semantic_score, alpha),
                **source["metadata"],
                bm25_score=bm25_score,
                semantic_score=semantic_score,
            )
        )
    return sorted(combined, key=lambda result: result["score"], reverse=True)


def rrf_score(rank: int, k: int = K_VALUE) -> float:
    if rank < 1 or k < 0:
        raise ValueError("rank must be positive and k cannot be negative")
    return 1 / (k + rank)


def rrf_combine_search_results(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    k: int = K_VALUE,
) -> list[SearchResult]:
    bm25_ranks = {result["id"]: rank for rank, result in enumerate(bm25_results, 1)}
    semantic_ranks = {result["id"]: rank for rank, result in enumerate(semantic_results, 1)}
    source_results = {result["id"]: result for result in bm25_results + semantic_results}
    fused: list[SearchResult] = []
    for chunk_id in bm25_ranks.keys() | semantic_ranks.keys():
        source = source_results[chunk_id]
        bm25_rank = bm25_ranks.get(chunk_id)
        semantic_rank = semantic_ranks.get(chunk_id)
        score = 0.0
        if bm25_rank is not None:
            score += rrf_score(bm25_rank, k)
        if semantic_rank is not None:
            score += rrf_score(semantic_rank, k)
        fused.append(
            format_search_result(
                doc_id=chunk_id,
                title=source["title"],
                document=source["document"],
                score=score,
                **source["metadata"],
                bm25_rank=bm25_rank,
                semantic_rank=semantic_rank,
            )
        )
    return sorted(fused, key=lambda result: result["score"], reverse=True)


def limit_chunks_per_parent(
    results: list[SearchResult],
    limit: int,
    max_per_parent: int = MAX_CHUNKS_PER_PARENT,
) -> list[SearchResult]:
    selected: list[SearchResult] = []
    counts: dict[str, int] = {}
    for result in results:
        parent_id = str(result["metadata"].get("parent_id") or result["id"])
        if counts.get(parent_id, 0) >= max_per_parent:
            continue
        selected.append(result)
        counts[parent_id] = counts.get(parent_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> WeightedSearchCommandResult:
    return {
        "original_query": query,
        "query": query,
        "alpha": alpha,
        "results": HybridSearch().weighted_search(query, alpha, limit),
    }


def rrf_search_command(
    query: str,
    k: int = K_VALUE,
    limit: int = DEFAULT_SEARCH_LIMIT,
    enhance: Literal["spell", "expand", "rewrite"] | None = None,
    rerank_method: Literal["individual", "batch", "cross_encoder"] | None = None,
) -> RRFSearchCommandResult:
    original_query = query
    enhanced_query = None
    if enhance:
        from .enhance_query import enhance_query

        enhanced_query = enhance_query(query, enhance)
        query = enhanced_query
    candidate_limit = limit * SEARCH_LIMIT_MULTIPLIER if rerank_method else limit
    print(f"Retrieving up to {candidate_limit} candidates for RRF search...")
    results = HybridSearch().rrf_search(query, k, candidate_limit)
    if rerank_method:
        from .rerank import rerank

        results = limit_chunks_per_parent(
            rerank(query, results, method=rerank_method, limit=len(results)), limit
        )
    return {
        "query": query,
        "k": k,
        "results": results,
        "enhance_method": enhance,
        "enhanced_query": enhanced_query,
        "rerank_method": rerank_method,
        "reranked": rerank_method is not None,
        "original_query": original_query,
    }
