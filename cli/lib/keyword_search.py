import math

from lib.inverted_index import InvertedIndex

from .config import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT
from .search_utils import load_chunks, tokenize_text, tokenize_term
import string


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    results = []
    try:
        index = InvertedIndex()
        index.load()
        query_tokens = tokenize_text(query)
        for token in query_tokens:
            doc_ids = index.get_documents(token)
            for doc_id in doc_ids:
                chunk = index.docmap[doc_id]
                if chunk not in results:
                    results.append(chunk)
                    if len(results) >= limit:
                        return results
    except FileNotFoundError as e:
        print(f"Error: {e}")
    return results

def token_appears_in_title(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def build_command() -> None:
    index = InvertedIndex()

    index.build(load_chunks())
    index.save()

    # docs = index.get_documents("merida")
    # print(f"First document for token 'merida' = {docs[0]}")

def tf_command(doc_id: str, term: str):
    try:
        index = InvertedIndex()
        index.load()
        tokenized_term = tokenize_term(term)
        frequency = index.get_term_frequency(doc_id, tokenized_term)
        print(frequency)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0

def idf_command(term: str):
    try:
        index = InvertedIndex()
        index.load()
        tokenized_term = tokenize_term(term)
        matching_docs = index.get_documents(tokenized_term)
        total_docs = len(index.docmap)
        idf_value = math.log((total_docs + 1) / (len(matching_docs) + 1))
        print(f"Inverse document frequency of '{term}': {idf_value:.2f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0.0

def tfidf_command(doc_id: str, term: str):
    try:
        index = InvertedIndex()
        index.load()
        tokenized_term = tokenize_term(term)
        tf_value = index.get_term_frequency(doc_id, tokenized_term)
        matching_docs = index.get_documents(tokenized_term)
        total_docs = len(index.docmap)
        idf_value = math.log((total_docs + 1) / (len(matching_docs) + 1))
        tfidf_value = tf_value * idf_value
        print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf_value:.2f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0.0

def bm25_idf_command(term: str) -> float:
    try:
        index = InvertedIndex()
        index.load()
        tokenized_term = tokenize_term(term)
        idf_value = index.get_bm25_idf(tokenized_term)
            
        return idf_value
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0.0

def bm25_tf_command(doc_id: str, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    try:
        index = InvertedIndex()
        index.load()
        tokenized_term = tokenize_term(term)
        tf_value = index.get_bm25_tf(doc_id, tokenized_term, k1=k1, b=b)
        return tf_value
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0.0

def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    try:
        index = InvertedIndex()
        index.load()
        res = index.bm25_search(query, limit)
        for result in res:
            print(f"({result['id']}) {result['title']} - Score: {result['score']:.2f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
