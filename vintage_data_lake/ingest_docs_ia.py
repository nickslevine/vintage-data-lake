import pathlib
import zstandard as zstd
from datetime import datetime
from blake3 import blake3
import polars as pl
from tqdm import tqdm
import uuid
import orjson
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa



BLOB_ROOT = pathlib.Path("/scratch/v13-ia-lake/data/blobs/blake3")
DOCS_OUT = "/scratch/v13-ia-lake/data/parquet/documents"


def norm_text(s: str) -> str:
    # normalize minimally; avoid destructive transforms now
    return s.replace("\r\n", "\n").strip()


def blob_path_for_digest(digest_hex: str) -> pathlib.Path:
    return BLOB_ROOT / digest_hex[:2] / digest_hex[2:4] / f"{digest_hex}.txt.zst"

def write_blob(text: str, digest_hex: str) -> int:
    p = blob_path_for_digest(digest_hex)
    p.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=10)
    data = text.encode("utf-8", errors="replace")
    with open(p, "wb") as f:
        with cctx.stream_writer(f) as zw:
            zw.write(data)
    return len(data)  # uncompressed size


def make_doc_row(rec, year, title, author, ingest_run_id: str):
    source = "ia"
    text = rec.get("text") or ""
    text_n = norm_text(text)
    digest_hex = blake3(text_n.encode("utf-8", "replace")).hexdigest()
    uncompressed_bytes = len(text_n.encode("utf-8", "replace"))
    # Prefer stable doc_id: hash(text + source_uri). Fallback to random.
    doc_id = rec.get("ia_id") or blake3((text_n + (rec.get("source_uri") or "")).encode()).hexdigest()
    # Blob URI
    text_uri = f"blob://blake3/{digest_hex[:2]}/{digest_hex[2:4]}/{digest_hex}.txt.zst"
    preview = text_n[:1024]
    now = datetime.now().isoformat(timespec="seconds") + "Z"
    return {
        "doc_id": doc_id,
        "source": source,
        "year": int(year) if year is not None else None,
        "title": title,
        "author": author,
        "text_uri": text_uri,
        "text_bytes": uncompressed_bytes,
        "text_checksum": digest_hex,
        "preview": preview,
        "ingest_run_id": ingest_run_id,
        "created_at": now,
        "updated_at": now,
        # keep the normalized text here temporarily; we'll drop before write
        "_normalized_text": text_n,
    }


def flush_rows(rows, ingest_run_id, write_i):
    # drop temp text and write to Parquet (partitioned)
    df = pl.DataFrame(rows).drop(["_normalized_text"])
    # ensure dtypes
    df = df.with_columns([
        pl.col("year").cast(pl.Int32, strict=False),
        pl.col("text_bytes").cast(pl.Int64),
    ])
    schema = pa.schema([
        pa.field("year", pa.int32()),
        pa.field("source", pa.string()),
    ])

    ds.write_dataset(
        data=df.to_arrow(),
        base_dir="/scratch/v13-ia-lake/data/parquet/documents",
        format="parquet",
        partitioning=ds.partitioning(schema, flavor="hive"),
        basename_template=f"part-{ingest_run_id}-{write_i}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore", 
    )


def ingest_jsonl(jsonl_path: str, batch_size: int = 10000):
    ingest_run_id = uuid.uuid4().hex
    rows = []
    count = 0
    metadata = pd.read_parquet("/data/ia-data/vintage_13_ia_index_with_metadata.parquet")
    metadata = metadata.set_index("ia_id")
    write_i = 0
    with open(jsonl_path, "rb") as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            rec = orjson.loads(line)
            ia_id = rec.get("ia_id")
            try:
                md = metadata.loc[ia_id]
            except Exception as _e:
                md = {}
            year = md.get("year") or 9999
            title = md.get("title")
            author = md.get("author")
            row = make_doc_row(rec, year, title, author, ingest_run_id)
            # write blob if missing
            digest = row["text_checksum"]
            bp = blob_path_for_digest(digest)
            if not bp.exists():
                write_blob(row["_normalized_text"], digest)
            rows.append(row)
            count += 1
            if len(rows) >= batch_size:
                flush_rows(rows, ingest_run_id, write_i)
                write_i += 1
                rows.clear()
    if rows:
        flush_rows(rows, ingest_run_id, write_i)
    print(f"ingested {count} docs into {DOCS_OUT}")

if __name__ == "__main__":
    ingest_jsonl("/data/ia-data/ocr_fixed_reconstructed.jsonl")