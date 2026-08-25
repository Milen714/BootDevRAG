import math

from lib.inverted_index import InvertedIndex

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, tokenize_text, tokenize_term
import string


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    # movies = load_movies()
    # results = []
    # query_tokens = tokenize_text(query)
    # for movie in movies:
    #     movie_tokens = tokenize_text(movie["title"])
    #     if token_appears_in_title(query_tokens, movie_tokens):
    #         results.append(movie)
    #         if len(results) >= limit:
    #             break
    # return results
    results = []
    try:
        index = InvertedIndex()
        index.load()
        query_tokens = tokenize_text(query)
        for token in query_tokens:
            doc_ids = index.get_documents(token)
            for doc_id in doc_ids:
                movie = index.docmap[doc_id]
                if movie not in results:
                    results.append(movie)
                    if len(results) >= limit:
                        return results
    except FileNotFoundError as e:
        print(f"Error: {e}")
    return results

def token_appears_in_title(query_tokens: list[str], movie_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True
    return False

def build_command() -> None:
    index = InvertedIndex()

    index.build()
    index.save()

    # docs = index.get_documents("merida")
    # print(f"First document for token 'merida' = {docs[0]}")

def tf_command(doc_id: int, term: str):
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

def tfidf_command(doc_id: int, term: str):
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