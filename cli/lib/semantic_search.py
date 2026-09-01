from typing import TypedDict

from sentence_transformers import SentenceTransformer
from .search_utils import CACHE_DIR,EMBEDDINGS_PATH , load_movies, DEFAULT_SEARCH_LIMIT
import numpy as np
import os



class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if len(text) == 0 or text == " ":
            raise ValueError("Input text cannot be empty or just whitespace.")
        input = [text]

        return self.model.encode(input)[0]

    def build_embeddings(self, documents) -> np.ndarray:
        if not documents:
            raise ValueError("Document list cannot be empty.")
        
        self.documents = documents

        movies_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            as_string = str(f"{doc['title']}: {doc['description']}")
            movies_strings.append(as_string)

        self.embeddings = self.model.encode(movies_strings, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        try:
            self.documents = documents
            self.document_map = {doc["id"]: doc for doc in documents}
            if os.path.exists(EMBEDDINGS_PATH):
                self.embeddings = np.load(EMBEDDINGS_PATH)
                if self.embeddings.shape[0] != len(documents):
                    print("Embeddings shape does not match the number of documents. Rebuilding embeddings...")
                    return self.build_embeddings(documents)
                else:
                    print("Embeddings loaded from cache.")
                    return self.embeddings
            print("Embeddings not found in cache. Building embeddings...")
            return self.build_embeddings(documents)
        except FileNotFoundError:
            print("Embeddings not found in cache. Building embeddings...")
            return self.build_embeddings(documents)

    def search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SemanticSearchResult]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities: list[tuple[float, Movie]] = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results: list[SemanticSearchResult] = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results



def verify_model():
    try:
        model = SemanticSearch().model
        print(f"Model loaded: {model}")
        print(f"Max sequence length: {model.max_seq_length}")
    except Exception as e:
        print(f"Error loading the model: {e}")

def embed_text(query: str):
    model = SemanticSearch()
    embedding = model.generate_embedding(query)
    print(f"Text: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    

def verify_embeddings():
    try:
        model = SemanticSearch()
        documents = load_movies()
        embeddings = model.load_or_create_embeddings(documents)
        print(f"Number of docs:   {len(documents)}")
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    except Exception as e:
        print(f"Error verifying embeddings: {e}")

        return None

def embed_query_text(query):
    model = SemanticSearch()
    embedding = model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
