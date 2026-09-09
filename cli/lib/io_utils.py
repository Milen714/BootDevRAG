import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def project_relative(path: Path) -> str:
    from .config import PROJECT_ROOT

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def records_hash(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        serialized = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest.update(serialized.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
