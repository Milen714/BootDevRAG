from typing import Any, TypedDict

from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
import re
from .search_utils import CACHE_DIR, CHUNK_OVERLAP, EMBEDDINGS_PATH, MODEL_NAME, load_movies, DEFAULT_SEARCH_LIMIT, CHUNK_SIZE, MAX_CHUNK_SIZE
import numpy as np
import os



class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class ChunkScore(TypedDict):
    movie_idx: int
    chunk_idx: int
    score: float


EmbeddingArray = NDArray[Any]


class SemanticSearch:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: EmbeddingArray | None = None
        self.documents: list[dict[str, Any]] | None = None
        self.document_map: dict[int, dict[str, Any]] = {}

    def generate_embedding(self, text: str) -> EmbeddingArray:
        if len(text) == 0 or text == " ":
            raise ValueError("Input text cannot be empty or just whitespace.")
        input = [text]

        return self.model.encode(input)[0]

    def build_embeddings(self, documents: list[dict[str, Any]]) -> EmbeddingArray:
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

    def load_or_create_embeddings(self, documents: list[dict[str, Any]]) -> EmbeddingArray:
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

        similarities: list[tuple[float, dict[str, Any]]] = []
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

def cosine_similarity(vec1: EmbeddingArray, vec2: EmbeddingArray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits the input text into chunks of specified size.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        raise ValueError("Input text cannot be empty.")
    
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")

    split_text = text.split()

    chunks = __chunk_text_impl(chunk_size, overlap, split_text)
            
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}\n")

    return chunks


def __chunk_text_impl(chunk_size, overlap, split_text)-> list[str]:
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk size.")

    for i in range(0, len(split_text), step):
        if i > 0 and len(split_text) - i <= overlap:
            break
        chunk = split_text[i:i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk.strip() for chunk in chunk if chunk.strip()))
    return chunks


def semantic_chunk_text(text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, verbose: bool = True) -> list[str]:
    """
    Splits the input text into semantic chunks of specified size.

    Args:
        text (str): The input text to be chunked.
        max_chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of semantic text chunks.
    """
    text = text.strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks: list[str] = []
    i = 0
    n_sentences = len(sentences)

    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            chunk_sentence = chunk_sentence.strip()
            if chunk_sentence:
                cleaned_sentences.append(chunk_sentence)
        if not cleaned_sentences:
            i += max_chunk_size - overlap
            continue
        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)
        i += max_chunk_size - overlap

    return chunks
