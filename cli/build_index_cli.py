import argparse

from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.config import CHUNKS_PATH, DATASET_PATH, PROCESSED_DOCUMENTS_PATH
from lib.ingestion import download_documents, parse_dataset
from lib.inverted_index import InvertedIndex
from lib.io_utils import read_jsonl
from lib.preprocessing import (
    chunk_cache_is_valid,
    chunk_documents,
    extract_and_clean_documents,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the document corpus and chunk-level search indexes"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build", action="store_true", help="Reuse processed chunks when available"
    )
    mode.add_argument(
        "--rebuild", action="store_true", help="Download, extract, and index everything again"
    )
    mode.add_argument(
        "--rechunk",
        action="store_true",
        help="Reuse extracted documents but regenerate chunks and indexes",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N Excel rows")
    args = parser.parse_args()

    if args.rechunk:
        if not PROCESSED_DOCUMENTS_PATH.exists():
            raise FileNotFoundError(
                f"Processed documents not found: {PROCESSED_DOCUMENTS_PATH}"
            )
        documents = read_jsonl(PROCESSED_DOCUMENTS_PATH)
        chunks = chunk_documents(documents)
        print(f"Regenerated {len(chunks)} retrieval chunks")
    elif (
        CHUNKS_PATH.exists()
        and PROCESSED_DOCUMENTS_PATH.exists()
        and not args.rebuild
        and args.limit is None
    ):
        documents = read_jsonl(PROCESSED_DOCUMENTS_PATH)
        if chunk_cache_is_valid(documents):
            chunks = read_jsonl(CHUNKS_PATH)
            print(f"Using {len(chunks)} processed chunks from {CHUNKS_PATH}")
        else:
            chunks = chunk_documents(documents)
            print(f"Chunk configuration changed; regenerated {len(chunks)} chunks")
    else:
        print(f"Reading dataset: {DATASET_PATH}")
        records = parse_dataset(limit=args.limit)
        print(f"Parsed {len(records)} document records")
        downloaded = download_documents(records, force=args.rebuild)
        ready = sum(
            record.get("status") in {"downloaded", "skipped_existing"}
            for record in downloaded
        )
        print(f"Sources ready: {ready}")
        documents = extract_and_clean_documents(downloaded)
        extracted = sum(
            document.get("extraction_status") == "extracted"
            for document in documents
        )
        print(f"Extracted {extracted} documents")
        chunks = chunk_documents(documents)
        print(f"Created {len(chunks)} retrieval chunks")

    if not chunks:
        raise RuntimeError(
            "No chunks were produced; inspect the download and extraction records"
        )

    print("Building BM25 index...")
    keyword_index = InvertedIndex()
    keyword_index.build(chunks)
    keyword_index.save()

    print("Building semantic embeddings...")
    ChunkedSemanticSearch().build_chunk_embeddings(chunks)
    print("Build complete")


if __name__ == "__main__":
    main()
