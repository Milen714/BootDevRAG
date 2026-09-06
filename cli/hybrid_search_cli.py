import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.hybrid_search import normalize_scores, rrf_search_command, weighted_search_command
from test_llm import spell_correction


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
        "--limit", type=int, default=10, help="Number of results to return"
    )
    rrf_search_parser.add_argument(
    "--enhance",
    type=str,
    choices=["spell"],
    help="Query enhancement method",
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
                print(f"   {res['document'][:100]}...")
                print()
        case "rrf-search":
            if args.enhance == "spell":
                method = "spell"
                query = args.query
                corrected_query = spell_correction(query)
                print(f"Enhanced query ({method}): '{query}' -> '{corrected_query}'\n")
                result = rrf_search_command(corrected_query, args.k, args.limit)
            else:
                result = rrf_search_command(args.query, args.k, args.limit)

            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
