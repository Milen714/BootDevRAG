import json
import os
import string
from typing import Any, TypedDict
from nltk.stem import PorterStemmer

class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]    

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 4
DOCUMENT_PREVIEW_LENGTH = 100

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNKED_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
JSON_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

CHUNK_SIZE = 200
CHUNK_OVERLAP = 0
MAX_CHUNK_SIZE = 4

BM25_K1 = 1.5
BM25_B = 0.75

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def format_search_result(doc_id, title, document, score, metadata=None) -> dict:
    return {
        "id": doc_id,
        "title": title,
        "document": document[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_stopwords() -> list[str]:

    words: list[str] = []
    with open(STOPWORDS_PATH, "r") as file:
        for line in file:
            word: str = line.strip()
            if word:
                words.append(preprocess_text(word))
    return words
STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    stemmer = PorterStemmer()
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token and not is_stopword(token):
            stemmed_token = stemmer.stem(token)
            valid_tokens.append(stemmed_token)
    return valid_tokens

def is_stopword(token: str) -> bool:
    return token in STOPWORDS

def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected a single token for term '{term}', but got {len(tokens)} tokens.")
    return tokens[0]
