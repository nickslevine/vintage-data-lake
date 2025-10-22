/scratch/v13-ia-lake/

```
 /data
  /raw/                       # your input JSONL lives here
  /blobs/                     # content-addressed text blobs (compressed)
  /parquet/
    documents/                # doc rows (no full text)
    doc_metadata/             # EAV extras (optional now)
    chunks/                   # (created later)
    embeddings/               # (later, per model)
    overlays/                 # (later)
  /registry/
    runs.parquet              # pipeline runs (later)
    models.parquet            # embedding models (later)
```