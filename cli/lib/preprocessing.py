import re
from functools import lru_cache
import json
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from transformers import AutoTokenizer

from .config import (
    CHUNKING_STRATEGY_VERSION,
    CHUNK_MANIFEST_PATH,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNKS_PATH,
    MIN_USEFUL_TEXT_CHARS,
    PROCESSED_DOCUMENTS_PATH,
    PROJECT_ROOT,
    REQUEST_TIMEOUT_SECONDS,
    TOKENIZER_NAME,
    USER_AGENT,
)
from .io_utils import project_relative, records_hash, write_json, write_jsonl


def clean_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = cleaned.replace("\x00", " ").replace("\ufeff", " ").replace("\u00ad", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"([A-Za-z])-\n\s*([A-Za-z])", r"\1\2", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+|[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def resolve_local_path(local_path: str | None) -> Path | None:
    if not local_path:
        return None
    path = Path(local_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_pdf_text(path: Path) -> tuple[str, dict[str, Any]]:
    reader = PdfReader(str(path))
    raw_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return raw_text, {
        "extraction_method": "pypdf",
        "page_count": len(reader.pages),
        "raw_text_char_count": len(raw_text),
    }


def extract_html_text(path: Path) -> tuple[str, dict[str, Any]]:
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "button"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    raw_text = main.get_text(separator="\n")
    return raw_text, {
        "extraction_method": "beautifulsoup4",
        "page_count": None,
        "raw_text_char_count": len(raw_text),
    }


def find_pdf_link(html: str, base_url: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        if ".pdf" in href.lower():
            return urljoin(base_url, href)
    return None


def extract_text_for_record(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    path = resolve_local_path(updated.get("local_path"))
    if updated.get("status") not in {"downloaded", "skipped_existing"} or path is None:
        updated.update(text="", text_char_count=0, extraction_status="skipped", extraction_error="No downloaded local source")
        return updated
    if not path.exists():
        updated.update(text="", text_char_count=0, extraction_status="extraction_failed", extraction_error=f"Local file not found: {path}")
        return updated
    try:
        if updated.get("source_type") == "pdf":
            raw_text, metadata = extract_pdf_text(path)
        elif updated.get("source_type") == "html":
            raw_text, metadata = extract_html_text(path)
            if len(clean_text(raw_text)) < MIN_USEFUL_TEXT_CHARS:
                pdf_url = find_pdf_link(
                    path.read_text(encoding="utf-8", errors="replace"),
                    str(updated.get("final_url") or updated.get("url") or ""),
                )
                if pdf_url:
                    response = requests.get(pdf_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    path = path.with_suffix(".pdf")
                    path.write_bytes(response.content)
                    raw_text, metadata = extract_pdf_text(path)
                    metadata["pdf_fallback_url"] = pdf_url
                    updated.update(local_path=project_relative(path), source_type="pdf")
        else:
            raise ValueError(f"Unsupported source type: {updated.get('source_type')}")
        text = clean_text(raw_text)
        status = "extracted" if len(text) >= MIN_USEFUL_TEXT_CHARS else ("empty_text" if not text else "too_short")
        updated.update(metadata)
        updated.update(
            text=text,
            text_char_count=len(text),
            extraction_status=status,
            extraction_error=None if status == "extracted" else f"Extracted text is shorter than {MIN_USEFUL_TEXT_CHARS} characters",
        )
    except Exception as error:
        updated.update(text="", text_char_count=0, extraction_status="extraction_failed", extraction_error=str(error))
    return updated


def extract_and_clean_documents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed = [extract_text_for_record(record) for record in records]
    write_jsonl(PROCESSED_DOCUMENTS_PATH, processed)
    return processed


def split_words(text: str) -> list[str]:
    return re.findall(r"\S+", text or "")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if part.strip()]


@lru_cache(maxsize=4)
def get_tokenizer(tokenizer_name: str = TOKENIZER_NAME) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    # Extraction text is counted before it is split, so counting must not warn or truncate.
    tokenizer.model_max_length = 10**9
    return tokenizer


def token_count(text: str, tokenizer: Any | None = None) -> int:
    active_tokenizer = tokenizer or get_tokenizer()
    return len(active_tokenizer.encode(text, add_special_tokens=False))


def split_token_windows(
    text: str, max_tokens: int, overlap_tokens: int, tokenizer: Any
) -> list[str]:
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = encoded["offset_mapping"]
    if len(offsets) <= max_tokens:
        return [text.strip()] if text.strip() else []
    windows: list[str] = []
    start = 0
    while start < len(offsets):
        end = min(start + max_tokens, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        window = text[char_start:char_end].strip()
        if window:
            windows.append(window)
        if end == len(offsets):
            break
        start = end - overlap_tokens
    return windows


def chunk_sentences(
    sentences: list[str],
    chunk_size: int = CHUNK_MAX_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
    tokenizer: Any | None = None,
) -> list[str]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "chunk_size must be positive and overlap must be between 0 and chunk_size"
        )
    active_tokenizer = tokenizer or get_tokenizer()
    units: list[tuple[str, int]] = []
    for sentence in sentences:
        for part in split_token_windows(
            sentence, chunk_size, overlap, active_tokenizer
        ):
            units.append((part, token_count(part, active_tokenizer)))

    chunks: list[str] = []
    start = 0
    while start < len(units):
        parts: list[str] = []
        used_tokens = 0
        end = start
        while end < len(units):
            unit_text, unit_tokens = units[end]
            if parts and used_tokens + unit_tokens > chunk_size:
                break
            parts.append(unit_text)
            used_tokens += unit_tokens
            end += 1
        if parts:
            chunk = " ".join(parts)
            actual_tokens = token_count(chunk, active_tokenizer)
            if actual_tokens > chunk_size:
                chunks.extend(
                    split_token_windows(
                        chunk, chunk_size, overlap, active_tokenizer
                    )
                )
            else:
                chunks.append(chunk)
        overlap_count = 0
        next_start = end
        for index in range(end - 1, start, -1):
            overlap_count += units[index][1]
            if overlap_count >= overlap:
                next_start = index
                break
        start = max(start + 1, next_start)
    return chunks


def create_chunks_for_document(
    document: dict[str, Any],
    chunk_size: int = CHUNK_MAX_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    if document.get("extraction_status") != "extracted":
        return []
    active_tokenizer = tokenizer or get_tokenizer()
    texts = chunk_sentences(
        split_sentences(str(document.get("text") or "")),
        chunk_size,
        overlap,
        active_tokenizer,
    )
    chunks: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        chunks.append(
            {
                "chunk_id": f"{document['doc_id']}::chunk-{index:04d}",
                "doc_id": str(document["doc_id"]),
                "title": document.get("title", ""),
                "text": text,
                "chunk_index": index,
                "total_chunks": len(texts),
                "chunk_word_count": len(split_words(text)),
                "chunk_token_count": token_count(text, active_tokenizer),
                **{
                    key: document.get(key, "")
                    for key in ("url", "publisher", "date", "theme", "keywords", "citation", "source_type", "local_path")
                },
            }
        )
    return chunks


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer = get_tokenizer()
    chunks = [
        chunk
        for document in documents
        for chunk in create_chunks_for_document(document, tokenizer=tokenizer)
    ]
    write_jsonl(CHUNKS_PATH, chunks)
    token_counts = [int(chunk["chunk_token_count"]) for chunk in chunks]
    sorted_counts = sorted(token_counts)
    write_json(
        CHUNK_MANIFEST_PATH,
        {
            "tokenizer": TOKENIZER_NAME,
            "chunking_strategy": CHUNKING_STRATEGY_VERSION,
            "chunk_max_tokens": CHUNK_MAX_TOKENS,
            "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
            "source_documents_hash": records_hash(documents),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "token_stats": {
                "minimum": min(token_counts, default=0),
                "average": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
                "median": sorted_counts[len(sorted_counts) // 2] if sorted_counts else 0,
                "p95": sorted_counts[int((len(sorted_counts) - 1) * 0.95)] if sorted_counts else 0,
                "maximum": max(token_counts, default=0),
                "over_limit": sum(count > CHUNK_MAX_TOKENS for count in token_counts),
            },
        },
    )
    return chunks


def chunk_cache_is_valid(documents: list[dict[str, Any]]) -> bool:
    if not CHUNKS_PATH.exists() or not CHUNK_MANIFEST_PATH.exists():
        return False
    try:
        manifest = json.loads(CHUNK_MANIFEST_PATH.read_text(encoding="utf-8"))
        return (
            manifest.get("tokenizer") == TOKENIZER_NAME
            and manifest.get("chunking_strategy") == CHUNKING_STRATEGY_VERSION
            and manifest.get("chunk_max_tokens") == CHUNK_MAX_TOKENS
            and manifest.get("chunk_overlap_tokens") == CHUNK_OVERLAP_TOKENS
            and manifest.get("source_documents_hash") == records_hash(documents)
        )
    except (OSError, ValueError):
        return False
