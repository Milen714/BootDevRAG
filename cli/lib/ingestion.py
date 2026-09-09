import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from .config import (
    DATASET_PATH,
    DOWNLOAD_DELAY_SECONDS,
    DOWNLOAD_REPORT_PATH,
    EXCEL_SHEET_NAME,
    PROCESSED_DOCUMENTS_PATH,
    RAW_DATA_DIR,
    REQUEST_TIMEOUT_SECONDS,
    REQUIRED_DATASET_COLUMNS,
    USER_AGENT,
)
from .io_utils import ensure_directory, project_relative, write_json, write_jsonl


def clean_optional_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return str(value.date()) if hasattr(value, "date") else value.isoformat()
    return " ".join(str(value).replace("\n", " ").split())


def load_dataset(
    dataset_path: Path = DATASET_PATH,
    sheet_name: str = EXCEL_SHEET_NAME,
    limit: int | None = None,
) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    dataframe = pd.read_excel(dataset_path, sheet_name=sheet_name)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        dataframe = dataframe.head(limit)
    return dataframe


def validate_dataset_columns(dataframe: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_DATASET_COLUMNS if c not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def infer_source_type(url: str, content_type: str | None = None) -> str:
    path = urlparse(url).path.lower()
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if media_type == "application/pdf":
        return "pdf"
    if media_type in {"text/html", "application/xhtml+xml"}:
        return "html"
    if path.endswith(".pdf"):
        return "pdf"
    if urlparse(url).scheme in {"http", "https"} and not media_type:
        return "web"
    return "unknown"


def normalize_documents(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    validate_dataset_columns(dataframe)
    records: list[dict[str, Any]] = []
    seen_urls: dict[str, str] = {}
    seen_ids: set[str] = set()
    for position, (_, row) in enumerate(dataframe.iterrows()):
        doc_id = clean_optional_string(row.get("id")) or f"DOC-{position + 1:04d}"
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate document id: {doc_id}")
        seen_ids.add(doc_id)
        url = clean_optional_string(row.get("url"))
        duplicate_of = seen_urls.get(url) if url else None
        if url and duplicate_of is None:
            seen_urls[url] = doc_id
        records.append(
            {
                "doc_id": doc_id,
                "title": clean_optional_string(row.get("title")),
                "url": url,
                "publisher": clean_optional_string(row.get("publisher")),
                "date": clean_optional_string(row.get("date")),
                "theme": clean_optional_string(row.get("theme")),
                "keywords": clean_optional_string(row.get("keywords")),
                "citation": clean_optional_string(row.get("bronvermelding")),
                "source_type": infer_source_type(url) if url else "missing",
                "local_path": None,
                "status": "duplicate" if duplicate_of else ("pending" if url else "missing_url"),
                "error": None if url else "Missing URL in dataset",
                "duplicate_of": duplicate_of,
            }
        )
    return records


def parse_dataset(
    dataset_path: Path = DATASET_PATH,
    sheet_name: str = EXCEL_SHEET_NAME,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records = normalize_documents(load_dataset(dataset_path, sheet_name, limit))
    write_jsonl(PROCESSED_DOCUMENTS_PATH, records)
    return records


def raw_file_path(doc_id: str, source_type: str, raw_dir: Path = RAW_DATA_DIR) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", doc_id).strip("._-") or "document"
    return raw_dir / f"{safe_id}{'.pdf' if source_type == 'pdf' else '.html'}"


def download_one_document(
    record: dict[str, Any],
    session: requests.Session,
    raw_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> dict[str, Any]:
    updated = dict(record)
    url = str(updated.get("url") or "")
    if updated.get("status") == "duplicate" or not url:
        return updated

    if not force:
        for cached_type in ("pdf", "html"):
            cached_path = raw_file_path(str(updated["doc_id"]), cached_type, raw_dir)
            if cached_path.exists() and cached_path.stat().st_size > 0:
                updated.update(
                    source_type=cached_type,
                    local_path=project_relative(cached_path),
                    status="skipped_existing",
                    error=None,
                )
                return updated

    source_type = infer_source_type(url)
    if source_type in {"web", "unknown"}:
        try:
            head = session.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT_SECONDS)
            updated["head_status_code"] = head.status_code
            source_type = infer_source_type(url, head.headers.get("content-type"))
        except requests.RequestException as error:
            updated["head_error"] = str(error)
    if source_type == "web":
        source_type = "html"

    ensure_directory(raw_dir)
    local_path = raw_file_path(str(updated["doc_id"]), source_type, raw_dir)
    updated.update(
        source_type=source_type,
        local_path=project_relative(local_path),
    )
    if local_path.exists() and local_path.stat().st_size > 0 and not force:
        updated.update(status="skipped_existing", error=None)
        return updated

    try:
        response: requests.Response | None = None
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = session.get(
                    url, allow_redirects=True, timeout=REQUEST_TIMEOUT_SECONDS
                )
                if response.status_code not in {202, 429} and response.content:
                    break
                retry_after = float(response.headers.get("Retry-After", 0) or 0)
                time.sleep(retry_after if retry_after > 0 else 2 * (attempt + 1))
            except requests.RequestException as error:
                last_error = error
                if attempt < 3:
                    time.sleep(2 * (attempt + 1))
        if response is None:
            raise last_error or requests.RequestException("No HTTP response received")
        updated.update(
            status_code=response.status_code,
            final_url=response.url,
            content_type=response.headers.get("content-type"),
        )
        response.raise_for_status()
        if not response.content:
            raise requests.RequestException(
                f"Source returned HTTP {response.status_code} with an empty body"
            )
        response_type = infer_source_type(response.url, response.headers.get("content-type"))
        if response_type == "web":
            response_type = "html"
        if response_type not in {"pdf", "html"}:
            updated.update(status="unsupported", error=f"Unsupported content type: {updated['content_type']}")
            return updated
        if response_type != source_type:
            source_type = response_type
            local_path = raw_file_path(str(updated["doc_id"]), source_type, raw_dir)
            updated.update(source_type=source_type, local_path=project_relative(local_path))
        local_path.write_bytes(response.content)
        updated.update(status="downloaded", error=None, bytes=local_path.stat().st_size)
    except (requests.RequestException, OSError) as error:
        updated.update(status="failed", error=str(error))
    return updated


def download_documents(
    records: list[dict[str, Any]], force: bool = False
) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    downloaded: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        downloaded.append(download_one_document(record, session, force=force))
        if position < len(records) - 1:
            time.sleep(DOWNLOAD_DELAY_SECONDS)
    write_jsonl(PROCESSED_DOCUMENTS_PATH, downloaded)
    failures = [
        {key: record.get(key) for key in ("doc_id", "title", "url", "status", "status_code", "error")}
        for record in downloaded
        if record.get("status") in {"failed", "unsupported", "missing_url"}
    ]
    write_json(
        DOWNLOAD_REPORT_PATH,
        {
            "total": len(downloaded),
            "downloaded": sum(r.get("status") == "downloaded" for r in downloaded),
            "cached": sum(r.get("status") == "skipped_existing" for r in downloaded),
            "failed": len(failures),
            "failures": failures,
        },
    )
    return downloaded
