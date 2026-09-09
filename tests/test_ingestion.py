from pathlib import Path

import pandas as pd

from lib.ingestion import infer_source_type, normalize_documents, validate_dataset_columns
from lib.preprocessing import (
    chunk_sentences,
    create_chunks_for_document,
    extract_html_text,
)


def dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "R01-01",
                "title": "Children and Deepfakes",
                "url": "https://example.test/source.pdf",
                "publisher": "Example Publisher",
                "date": pd.Timestamp("2025-07-01"),
                "theme": "Deepfakes",
                "keywords": "children, safety",
                "bronvermelding": "Example citation",
            }
        ]
    )


def test_normalize_documents_preserves_source_metadata() -> None:
    records = normalize_documents(dataset_frame())

    assert records == [
        {
            "doc_id": "R01-01",
            "title": "Children and Deepfakes",
            "url": "https://example.test/source.pdf",
            "publisher": "Example Publisher",
            "date": "2025-07-01",
            "theme": "Deepfakes",
            "keywords": "children, safety",
            "citation": "Example citation",
            "source_type": "pdf",
            "local_path": None,
            "status": "pending",
            "error": None,
            "duplicate_of": None,
        }
    ]


def test_dataset_validation_lists_missing_columns() -> None:
    frame = dataset_frame().drop(columns=["url", "theme"])

    try:
        validate_dataset_columns(frame)
    except ValueError as error:
        assert "url" in str(error)
        assert "theme" in str(error)
    else:
        raise AssertionError("Expected missing-column validation to fail")


def test_duplicate_document_ids_are_rejected() -> None:
    frame = pd.concat([dataset_frame(), dataset_frame()], ignore_index=True)

    try:
        normalize_documents(frame)
    except ValueError as error:
        assert "Duplicate document id" in str(error)
    else:
        raise AssertionError("Expected duplicate IDs to fail")


def test_response_content_type_overrides_url_extension() -> None:
    assert infer_source_type("https://example.test/file.pdf", "text/html") == "html"


def test_html_extraction_removes_navigation(tmp_path: Path) -> None:
    path = tmp_path / "source.html"
    path.write_text(
        "<html><nav>Menu text</nav><main><h1>Policy</h1><p>Relevant evidence.</p></main></html>",
        encoding="utf-8",
    )

    text, metadata = extract_html_text(path)

    assert "Relevant evidence" in text
    assert "Menu text" not in text
    assert metadata["extraction_method"] == "beautifulsoup4"


def test_sentence_chunks_overlap_and_keep_parent_metadata() -> None:
    sentences = [f"Sentence {index} has five useful words." for index in range(8)]
    texts = chunk_sentences(sentences, chunk_size=20, overlap=5)
    document = {
        "doc_id": "R01-01",
        "title": "Policy",
        "text": " ".join(sentences),
        "extraction_status": "extracted",
        "url": "https://example.test/policy",
        "publisher": "Publisher",
    }
    chunks = create_chunks_for_document(document, chunk_size=20, overlap=5)

    assert len(texts) > 1
    assert chunks[0]["chunk_id"] == "R01-01::chunk-0000"
    assert chunks[0]["doc_id"] == "R01-01"
    assert chunks[0]["url"] == document["url"]
    assert chunks[0]["text"] != document["text"]
    assert set(texts[0].split()) & set(texts[1].split())
