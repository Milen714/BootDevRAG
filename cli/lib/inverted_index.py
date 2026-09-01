import math

from .search_utils import tokenize_text, load_movies, BM25_K1, CACHE_DIR, BM25_B
import os 
import pickle
from collections import Counter

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
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

        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl"):
            raise FileNotFoundError("Inverted index files not found. Please build the index first.")
        
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)

        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_term_frequency(self, doc_id, term) -> int:
        if doc_id in self.term_frequencies:
            return self.term_frequencies.get(doc_id, Counter())[term]
        return 0

    def get_bm25_idf(self, term: str) -> float:
        """
        Calculate the Inverse Document Frequency (IDF) for a given term using the BM25 formula.
        IDF = log((N - n + 0.5) / (n + 0.5) + 1)
        where:
            N = total number of documents
            n = number of documents containing the term
        """
        N = len(self.docmap)

        df = len(self.index.get(term, set()))
        IDF = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return IDF

    def __get_avg_doc_length(self) -> float:
        """
        Calculate the average document length across all documents in the index.
        """
        total_length = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)
        return total_length / num_docs if num_docs > 0 else 0.0
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        """
        Calculate the Term Frequency (TF) for a given term in a specific document using the BM25 formula.
        TF = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        where:
            tf = term frequency in the document
            k1 = BM25 parameter (default is 1.5)
            b = BM25 parameter (default is 0.75)
            length_norm = 1 - b + b * (document length / average document length)
        """
        tf = self.get_term_frequency(doc_id, term)
        if tf == 0:
            return 0.0
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf

    def bm25(self, doc_id, term):
        """
        Calculate the BM25 score for a given term in a specific document.
        BM25 = IDF * TF
        where:
            IDF = Inverse Document Frequency for the term
            TF = Term Frequency for the term in the document
        """
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return idf * tf
    def bm25_search(self, query, limit) -> list[tuple[dict, float]]:
        """
        Perform a BM25 search for a given query and return the top results.
        """
        tokens = tokenize_text(query)
        scores = {}
        for token in tokens:
            for doc_id in self.get_documents(token):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += self.bm25(doc_id, token)
        
        # Sort documents by score in descending order and limit the results
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [(self.docmap[doc_id], score) for doc_id, score in sorted_docs]
