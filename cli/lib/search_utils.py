import string
from typing import Any, TypedDict
from nltk.stem import PorterStemmer

from .config import CHUNKS_PATH, SCORE_PRECISION, STOPWORDS_PATH
from .io_utils import read_jsonl

class SearchResult(TypedDict):
    id: str
    title: str
    document: str
    score: float
    metadata: dict[str, Any]    

def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk dataset not found at {CHUNKS_PATH}. Run the build command first."
        )
    return read_jsonl(CHUNKS_PATH)


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_stopwords() -> list[str]:

    words: list[str] = []
    with STOPWORDS_PATH.open("r", encoding="utf-8") as file:
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
