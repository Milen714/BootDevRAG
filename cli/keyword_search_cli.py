import argparse
from lib.search_utils import BM25_B, BM25_K1
from lib.keyword_search import bm25_tf_command, bm25search_command, search_command, build_command, tf_command, idf_command, tfidf_command, bm25_idf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a specific document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a specific term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for a specific document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")    

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter") 
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")  

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            print(f"Getting term frequency for doc_id: {args.doc_id}, term: {args.term}")
            tf_command(args.doc_id, args.term)
        case "idf":
            print(f"Getting inverse document frequency for term: {args.term}")
            idf_command(args.term)
        case "tfidf":
            print(f"Getting TF-IDF for doc_id: {args.doc_id}, term: {args.term}")
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            print(f"Getting BM25 IDF for term: {args.term}")
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            print(f"Getting BM25 TF for doc_id: {args.doc_id}, term: {args.term}, k1: {args.k1}, b: {args.b}")
            bm25tf = bm25_tf_command(args.doc_id, args.term, k1=args.k1, b=args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            print(f"Searching for: {args.query} using BM25 scoring")
            bm25search_command(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()