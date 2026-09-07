import os
from typing import Literal, TypedDict

from lib.search_utils import DEFAULT_SEARCH_LIMIT, DOCUMENT_PREVIEW_LENGTH, DEFAULT_ALPHA, K_VALUE, SEARCH_LIMIT_MULTIPLIER, SearchResult, load_movies, format_search_result
from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch


class CombinedScoreData(TypedDict):
    title: str
    document: str
    bm25_score: float
    semantic_score: float


class RRFScoreData(TypedDict):
    title: str
    document: str
    bm25_rank: int | None
    semantic_rank: int | None
    rrf_score: float

class RRFSearchCommandResult(TypedDict):
    original_query: str
    enhanced_query: str | None
    enhance_method: Literal["spell", "expand", "rewrite"] | None
    query: str
    k: int
    rerank_method: Literal["individual", "batch"] | None
    reranked: bool
    results: list[SearchResult]


class WeightedSearchCommandResult(TypedDict):
    original_query: str
    query: str
    alpha: float
    results: list[SearchResult]

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        bm25_results = self._bm25_search(query, DEFAULT_SEARCH_LIMIT)
        semantic_results = self.semantic_search.search_chunks(query, DEFAULT_SEARCH_LIMIT)

        combined_results = combine_search_results(bm25_results, semantic_results, alpha)


        return combined_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined_results = rff_combine_search_results(bm25_results, semantic_results, k)
        return combined_results[:limit]

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA) -> float:
        return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))
    return normalized_scores

def normalize_search_results(
    results: list[SearchResult],
) -> list[tuple[SearchResult, float]]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    return list(zip(results, normalized))

def combine_search_results(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    alpha: float = DEFAULT_ALPHA,
) -> list[SearchResult]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores: dict[int, CombinedScoreData] = {}

    for result, normalized_score in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if normalized_score > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = normalized_score

    for result, normalized_score in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if normalized_score > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = normalized_score

    hybrid_results: list[SearchResult] = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def rff_combine_search_results(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    k: int = K_VALUE,
) -> list[SearchResult]:
    combined_scores: dict[int, RRFScoreData] = {}

    for rank, result in enumerate(bm25_results, 1):
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": None,
                "semantic_rank": None,
                "rrf_score": 0.0,
            }
        if combined_scores[doc_id]["bm25_rank"] is None:
            combined_scores[doc_id]["bm25_rank"] = rank
            combined_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    for rank, result in enumerate(semantic_results, 1):
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": None,
                "semantic_rank": None,
                "rrf_score": 0.0,
            }
        if combined_scores[doc_id]["semantic_rank"] is None:
            combined_scores[doc_id]["semantic_rank"] = rank
            combined_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    rrf_results: list[SearchResult] = []
    for doc_id, data in combined_scores.items():
        rrf_results.append(
            format_search_result(
                doc_id=doc_id,
                title=data["title"],
                document=data["document"],
                score=data["rrf_score"],
                bm25_rank=data["bm25_rank"],
                semantic_rank=data["semantic_rank"],
            )
        )

    return sorted(rrf_results, key=lambda x: x["score"], reverse=True)


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> WeightedSearchCommandResult:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }
def rrf_search_command(
    query: str, k: int = K_VALUE, limit: int = DEFAULT_SEARCH_LIMIT,
    enhance: Literal["spell", "expand", "rewrite"] | None = None,
    rerank_method: Literal["individual", "batch", "cross_encoder"] | None = None,
) -> RRFSearchCommandResult:
    movies = load_movies()
    searcher = HybridSearch(movies)
    original_query = query
    enhanced_query = None

    if enhance:
        from .enhance_query import enhance_query

        enhanced_query = enhance_query(query, enhance)
        query = enhanced_query

    search_limit = (
        limit * SEARCH_LIMIT_MULTIPLIER
        if rerank_method
        else limit
    )

    results = searcher.rrf_search(query, k, search_limit)

    reranked = False
    if rerank_method:
        from .rerank import rerank

        results = rerank(query, results, method=rerank_method, limit=limit)
        reranked = True
    return {
        "query": query,
        "k": k,
        "results": results,
        "enhance_method": enhance,
        "enhanced_query": enhanced_query,
        "rerank_method": rerank_method,
        "reranked": reranked,
        "original_query": original_query,
        }
