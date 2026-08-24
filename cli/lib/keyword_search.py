from lib.inverted_index import InvertedIndex

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, tokenize_text
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
    