import argparse
import time

from lib.config import DEFAULT_SEARCH_LIMIT, DOCUMENT_PREVIEW_LENGTH
from lib.hybrid_search import normalize_scores, rrf_search_command, weighted_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
     

    # Add subparsers for each command
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="*", type=float, help="List of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform a weighted search"
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="Query to search for"
    )
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for the semantic search"
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform a Reciprocal Rank Fusion (RRF) search"
    )
    rrf_search_parser.add_argument(
        "query", type=str, help="Query to search for"
    )
    rrf_search_parser.add_argument(
        "--k", type=int, default=60, help="Number of top results to consider for RRF"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return"
    )
    rrf_search_parser.add_argument(
    "--enhance",
    type=str,
    choices=["spell", "rewrite", "expand"],
    help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method to use after RRF",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                metadata = res.get("metadata", {})
                print(f"   Chunk: {res['id']} (parent: {metadata.get('parent_id', 'unknown')})")
                print(f"   Source: {metadata.get('publisher', '')} {metadata.get('date', '')}".rstrip())
                if metadata.get("url"):
                    print(f"   URL: {metadata['url']}")
                print(f"   {res['document'][:DOCUMENT_PREVIEW_LENGTH]}...")
                print()
        case "rrf-search":
            result = rrf_search_command(
                query=args.query,
                k=args.k,
                limit=args.limit,
                enhance=args.enhance,
                rerank_method=args.rerank_method,
            )

            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )

            if result["reranked"]:
                print(
                    f"Re-ranking top {len(result['results'])} results using {result['rerank_method']} method...\n"
                )

            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Re-rank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Re-rank Rank: {res.get('batch_rank', 0)}")
                if "crossencoder_score" in res:
                    print(f"   Cross-encoder Score: {float(res['crossencoder_score']):.3f}")
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   Chunk: {res['id']} (parent: {metadata.get('parent_id', 'unknown')})")
                print(f"   Source: {metadata.get('publisher', '')} {metadata.get('date', '')}".rstrip())
                if metadata.get("url"):
                    print(f"   URL: {metadata['url']}")
                if metadata.get("citation"):
                    print(f"   Citation: {metadata['citation']}")
                print(f"   {res['document']}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
