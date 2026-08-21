from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    translator = str.maketrans('', '', string.punctuation)
    for movie in movies:
        query = query.lower().translate(translator)
        if query in movie["title"].lower().translate(translator):
            results.append(movie)
            if len(results) >= limit:
                break
    return results