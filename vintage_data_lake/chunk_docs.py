import pyarrow as pa
import pyarrow.dataset as ds
import io

from datetime import datetime, timezone
import zstandard as zstd
import blake3
import re
import os
import tiktoken
from dataclasses import dataclass

BLOB_URI_RE = re.compile(r"^blob://blake3/([0-9a-f]{2})/([0-9a-f]{2})/([0-9a-f]{64})\\.txt\\.zst$")

@dataclass
class Paths:
    base: str

    @property
    def documents(self) -> str:
        return os.path.join(self.base, "parquet/documents")

    @property
    def chunks(self) -> str:
        return os.path.join(self.base, "parquet/chunks")

    @property
    def blobs(self) -> str:
        return os.path.join(self.base, "blobs")



class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, toks: list[int]) -> str:
        return self.enc.decode(toks)

def now_ts() -> datetime:
    return datetime.now(timezone.utc)

def b3(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()

def b3_txt(s: str) -> str:
    return b3(s.encode("utf-8"))

def blob_path_from_uri(base_blobs: str, uri: str) -> str:
    m = BLOB_URI_RE.match(uri)
    if not m:
      raise ValueError(f"Unsupported blob uri: {uri}")
    aa, bb, digest = m.groups()
    return os.path.join(base_blobs, "blake3", aa, bb, f"{digest}.txt.zst")


def read_blob(uri: str, base_blobs: str) -> str:
    p = os.path.join(base_blobs, uri[7:])
    dctx = zstd.ZstdDecompressor()

    with open(p, "rb") as f:
        with dctx.stream_reader(f) as r:
            with io.TextIOWrapper(r, encoding="utf-8") as s:
                return s.read()


# Soft sentence boundaries to avoid chopping mid-sentence (optional)
SENT_SPLIT = re.compile(r"(?<=[.!?])(\s+|\n+)\n?")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in SENT_SPLIT.finditer(text):
        end = m.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans

def merge_spans_to_token_windows(tokens, window: int, overlap: int):

    n = len(tokens)
    if n == 0:
        return [(0, 0)]

    if window <= 0:
        raise ValueError("window must be > 0")
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must satisfy 0 <= overlap < window")

    spans = []
    stride = window - overlap
    i = 0
    while i < n:
        j = min(n, i + window)
        spans.append((i, j))
        if j == n:
            break
        i += stride
    return spans



def make_chunk_rows(doc_row, text: str, tk: Tokenizer, window: int, overlap: int) -> list[dict]:
    doc_id = doc_row["doc_id"]
    year = int(doc_row["year"]) if doc_row["year"] is not None else None
    source = doc_row["source"] or ""
    ingest_run_id = doc_row["ingest_run_id"] or ""

    toks = tk.encode(text)
    spans = merge_spans_to_token_windows(toks, window, overlap)
    rows: list[dict] = []
    for seq, (s, e) in enumerate(spans):
        chunk_toks = toks[s:e]
        chunk_txt = tk.decode(chunk_toks)
        row = {
            "chunk_id": b3_txt(f"{doc_id}:{seq}:{s}:{e}:{b3_txt(chunk_txt)}"),
            "doc_id": doc_id,
            "year": year,
            "source": source,
            "seq": seq,
            "start_token": s,
            "end_token": e,
            "text": chunk_txt,
            "text_checksum": b3_txt(chunk_txt),
            "token_count": len(chunk_toks),
            "ingest_run_id": ingest_run_id,
            "created_at": now_ts(),
            "tokens": tk.encode(chunk_txt),
        }
        rows.append(row)
    return rows


def rows_to_table(rows: list[dict]) -> pa.Table:
    arrays = {
        "chunk_id": pa.array([r["chunk_id"] for r in rows], pa.string()),
        "doc_id": pa.array([r["doc_id"] for r in rows], pa.string()),
        "year": pa.array([r["year"] for r in rows], pa.int32()),
        "source": pa.array([r["source"] for r in rows], pa.string()),
        "seq": pa.array([r["seq"] for r in rows], pa.int32()),
        "start_token": pa.array([r["start_token"] for r in rows], pa.int32()),
        "end_token": pa.array([r["end_token"] for r in rows], pa.int32()),
        "text": pa.array([r["text"] for r in rows], pa.string()),
        "text_checksum": pa.array([r["text_checksum"] for r in rows], pa.string()),
        "token_count": pa.array([r["token_count"] for r in rows], pa.int32()),
        "ingest_run_id": pa.array([r["ingest_run_id"] for r in rows], pa.string()),
        "created_at": pa.array([r["created_at"] for r in rows], pa.timestamp("us")),
        "tokens": pa.array([r["tokens"] for r in rows], pa.list_(pa.int32())),
    }
    return pa.table(arrays)


def write_chunks(paths: Paths, table: pa.Table, run_id: str, write_i: int) -> None:
    basename_template = f"part-{run_id}-{write_i}-{{i}}.parquet"

    schema = pa.schema([
        pa.field("year", pa.int32()),
        pa.field("source", pa.string()),
    ])
    ds.write_dataset(
        data=table,
        base_dir=paths.chunks,
        format="parquet",
        partitioning=ds.partitioning(schema, flavor="hive"),
        existing_data_behavior="error",  # safe because basenames are unique per run
        basename_template=basename_template,
    )



def run(base: str, window: int = 512, overlap: int = 64) -> None:
    paths = Paths(base)
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    tk = Tokenizer()

    dataset = ds.dataset(paths.documents, format="parquet", partitioning="hive")
    # Select required columns to minimize IO
    cols = ["doc_id", "year", "source", "text_uri", "ingest_run_id"]
    scan = dataset.scanner(columns=cols)

    written = 0
    batch_rows = 0
    out_rows: list[dict] = []
    write_i = 0

    for batch in scan.to_batches():
        batch_table = pa.Table.from_batches([batch])
        for row in batch_table.to_pylist():
            # row is dict[str, Any]
            try:
                text = read_blob(row["text_uri"], paths.blobs)
            except FileNotFoundError as e:
                print(f"File not found for row: {row['doc_id']}, {row['text_uri']}, {e}")
                continue
            rows = make_chunk_rows(row, text, tk, window, overlap)
            out_rows.extend(rows)
            if len(out_rows) >= 20000:
                tbl = rows_to_table(out_rows)
                write_chunks(paths, tbl, run_id, write_i)
                write_i += 1
                written += len(out_rows)
                out_rows.clear()
                print(f"[flush] total rows written: {written}")
            batch_rows += 1

    if out_rows:
        tbl = rows_to_table(out_rows)
        write_chunks(paths, tbl, run_id, write_i)
        written += len(out_rows)
        print(f"[flush] total rows written: {written}")
        out_rows.clear()

if __name__ == "__main__":
    run("/scratch/v13-ia-lake/data/")