import argparse

from lib.semantic_search import SemanticSearch, embed_query_text, verify_embeddings, verify_model, embed_text
from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_model_parser = subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings for the documents")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding for a given query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to generate embedding for")

    search_parser = subparsers.add_parser("search", help="Search documents using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Verifying semantic search model...")
            verify_model()
        case "embed_text":
            print(f"Generating embedding for text: {args.text}")
            embed_text(args.text)  
        case "verify_embeddings":
            print("Verifying embeddings for the documents...")
            verify_embeddings()
        case "embed_query":
            print(f"Generating embedding for query: {args.query}")
            embed_query_text(args.query)
        case "search":
            model = SemanticSearch()
            documents = load_movies()
            model.load_or_create_embeddings(documents)
            results = model.search(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"  {res['description']}\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
