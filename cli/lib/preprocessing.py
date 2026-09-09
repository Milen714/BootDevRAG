import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import (
    CHUNKS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_USEFUL_TEXT_CHARS,
    PROCESSED_DOCUMENTS_PATH,
    PROJECT_ROOT,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)
from .io_utils import project_relative, write_jsonl


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


def chunk_sentences(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be between 0 and chunk_size")
    sentence_words = [split_words(sentence) for sentence in sentences]
    chunks: list[str] = []
    start = 0
    while start < len(sentence_words):
        words: list[str] = []
        end = start
        while end < len(sentence_words):
            candidate = words + sentence_words[end]
            if words and len(candidate) > chunk_size:
                break
            words = candidate
            end += 1
        if words:
            chunks.append(" ".join(words))
        overlap_words = 0
        next_start = end
        for index in range(end - 1, start, -1):
            overlap_words += len(sentence_words[index])
            if overlap_words >= overlap:
                next_start = index
                break
        start = max(start + 1, next_start)
    return chunks


def create_chunks_for_document(
    document: dict[str, Any], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[dict[str, Any]]:
    if document.get("extraction_status") != "extracted":
        return []
    texts = chunk_sentences(split_sentences(str(document.get("text") or "")), chunk_size, overlap)
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
                **{
                    key: document.get(key, "")
                    for key in ("url", "publisher", "date", "theme", "keywords", "citation", "source_type", "local_path")
                },
            }
        )
    return chunks


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks = [chunk for document in documents for chunk in create_chunks_for_document(document)]
    write_jsonl(CHUNKS_PATH, chunks)
    return chunks
