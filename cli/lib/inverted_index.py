from .search_utils import tokenize_text, load_movies
import os 
import pickle

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        if term in self.index:
            return sorted(list(self.index[term]))
        return []

    def build(self):
        movies = load_movies()

        for movie in movies:

            doc_id = movie["id"]
            self.docmap[doc_id] = movie

            combined_text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, combined_text)

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl"):
            raise FileNotFoundError("Inverted index files not found. Please build the index first.")
        
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)

        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        