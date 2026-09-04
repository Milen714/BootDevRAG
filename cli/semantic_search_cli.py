import argparse

from lib.semantic_search import SemanticSearch, chunk_text, embed_query_text, semantic_chunk_text, verify_embeddings, verify_model, embed_text
from lib.search_utils import DEFAULT_SEARCH_LIMIT, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNK_SIZE, load_movies
from lib.chunked_semantic_search import ChunkedSemanticSearch

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

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search documents using chunked semantic search")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Create chunks")
    chunk_parser.add_argument("text", type=str, help="Text to create chunks for")
    chunk_parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Overlap size between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Create semantic chunks")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to create semantic chunks for")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=MAX_CHUNK_SIZE, help="Size of each semantic chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Overlap size between semantic chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate embeddings for semantic chunks")

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
        case "search_chunked":
            model = ChunkedSemanticSearch()
            documents = load_movies()
            model.load_or_create_chunk_embeddings(documents)
            results = model.search_chunks(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")
        case "chunk":
            print(f"Creating chunks for text: {args.text}")
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            print(f"Creating semantic chunks for text: {args.text}")
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            print("Generating embeddings for semantic chunks...")
            model = ChunkedSemanticSearch()
            documents = load_movies()
            embeddings = model.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
