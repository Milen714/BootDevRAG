## BootDevRAG

BootDevRAG builds a hybrid chunk-retrieval index from the policy-document Excel
catalog at `data/20260331_dataset2.xlsx`. It downloads PDF and HTML sources,
extracts and cleans their text, creates sentence-aware chunks, and indexes the
same chunks for BM25 and semantic cosine-similarity search.

### Setup

```bash
uv sync --dev
```

The source workbook is project-local but `data/` is ignored because it also
contains downloaded and processed documents.

### Build

Build from Excel, reusing downloaded files and processed chunks when possible:

```bash
uv run python cli/build_index_cli.py --build
```

Force downloading, extraction, chunking, and indexing again:

```bash
uv run python cli/build_index_cli.py --rebuild
```

Use `--limit N` for a smaller ingestion run. Generated parent documents and
chunks are stored under `data/processed/`; BM25 and embedding artifacts are
stored under `cache/`. Both indexes include corpus hashes and are rebuilt when
their chunk corpus or configuration changes.

### Search

```bash
uv run python cli/hybrid_search_cli.py rrf-search "children and deepfakes" --limit 5
uv run python cli/hybrid_search_cli.py rrf-search "children and deepfakes" --limit 5 --rerank-method cross_encoder
uv run python cli/hybrid_search_cli.py weighted-search "AI literacy" --limit 5
```

Results contain only relevant chunks. Each result has a stable string
`chunk_id`, a `parent_id`, chunk position, URL, publisher, date, theme,
keywords, citation, and retrieval ranks. At most two final chunks are returned
from one parent document.

### Tests

```bash
uv run pytest -q
```
