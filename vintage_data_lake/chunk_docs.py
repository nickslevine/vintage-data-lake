import pyarrow as pa
import pyarrow.dataset as ds
from datetime import datetime, timezone
import zstandard as zstd
import blake3
import re
import os
from dataclasses import dataclass
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import threading
from queue import Queue

BLOB_URI_RE = re.compile(r"^blob://blake3/([0-9a-f]{2})/([0-9a-f]{2})/([0-9a-f]{64})\\.txt\\.zst$")

_worker_tokenizer = None
_worker_decompressor = None

def _init_worker() -> None:
    """Initialize worker process with tokenizer and decompressor."""
    global _worker_tokenizer, _worker_decompressor
    _worker_tokenizer = Tokenizer()
    _worker_decompressor = zstd.ZstdDecompressor()

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
        self.tk = AutoTokenizer.from_pretrained("intfloat/e5-large-v2", use_fast=True)

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text)

    def decode(self, toks: list[int]) -> str:
        return self.tk.decode(toks, skip_special_tokens=True, clean_up_tokenization_spaces=True)

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


def read_blob(uri: str, base_blobs: str, dctx: zstd.ZstdDecompressor | None = None) -> str:
    """Read and decompress a blob. Reuses decompressor if provided for efficiency."""
    p = os.path.join(base_blobs, uri[7:])
    if dctx is None:
        dctx = zstd.ZstdDecompressor()
    
    # Read entire file at once for better I/O performance
    with open(p, "rb") as f:
        compressed_data = f.read()
    
    # Use streaming reader to handle files without content size in header
    decompressed = dctx.stream_reader(compressed_data).read()
    return decompressed.decode("utf-8")


def merge_spans_to_token_windows(tokens: list[int], window: int, overlap: int) -> list[tuple[int, int]]:

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



def make_chunk_rows(doc_row: dict, text: str, tk: Tokenizer, window: int, overlap: int) -> list[dict]:
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


def _process_row(row: dict, base_blobs: str, window: int, overlap: int) -> list[dict]:
    """Process a single document row: read blob, tokenize, create chunks."""
    assert _worker_tokenizer is not None
    assert _worker_decompressor is not None
    text = read_blob(row["text_uri"], base_blobs, _worker_decompressor)
    rows = make_chunk_rows(row, text, _worker_tokenizer, window, overlap)
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
    """Write chunks table to partitioned parquet dataset."""
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
        existing_data_behavior="overwrite_or_ignore",  # safe because basenames are unique per run
        basename_template=basename_template,
    )


class AsyncWriter:
    """Asynchronous writer to avoid blocking processing on I/O."""
    
    def __init__(self, paths: Paths, run_id: str):
        self.paths = paths
        self.run_id = run_id
        self.queue: Queue = Queue(maxsize=3)
        self.write_i = 0
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
    
    def _writer_loop(self):
        """Background thread that writes tables to disk."""
        while True:
            item = self.queue.get()
            if item is None:  # Sentinel to stop
                break
            table, write_i = item
            write_chunks(self.paths, table, self.run_id, write_i)
            self.queue.task_done()
    
    def write(self, table: pa.Table) -> int:
        """Queue a table for writing. Returns the write index."""
        write_i = self.write_i
        self.write_i += 1
        self.queue.put((table, write_i))
        return write_i
    
    def close(self):
        """Wait for all pending writes to complete."""
        self.queue.join()
        self.queue.put(None)  # Sentinel
        self.thread.join(timeout=30)



def run(base: str, window: int = 512, overlap: int = 64, prefetch_batches: int = 200) -> None:
    """
    Chunk documents into overlapping token windows with streaming pipeline.
    
    Args:
        base: Base directory path containing documents and blobs
        window: Token window size for chunks
        overlap: Token overlap between chunks
        prefetch_batches: Number of document batches to keep in flight (should be > num_workers)
    """
    paths = Paths(base)
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")

    dataset = ds.dataset(paths.documents, format="parquet", partitioning="hive")
    # Select required columns to minimize IO
    cols = ["doc_id", "year", "source", "text_uri", "ingest_run_id"]
    scan = dataset.scanner(columns=cols)

    written = 0
    out_rows: list[dict] = []

    pbar = tqdm(desc="Chunking documents", unit=" chunks")

    process_fn = partial(_process_row, base_blobs=paths.blobs, window=window, overlap=overlap)
    
    # Use all available cores
    pool = mp.Pool(processes=100, initializer=_init_worker, maxtasksperchild=1000)
    
    # Use async writer to avoid blocking on writes
    writer = AsyncWriter(paths, run_id)
    
    # Generator to yield all rows from dataset
    def row_generator():
        for batch in scan.to_batches():
            batch_table = pa.Table.from_batches([batch])
            for row in batch_table.to_pylist():
                yield row
    
    # Use imap_unordered for streaming pipeline - workers process continuously
    # chunksize controls how many tasks are sent to each worker at once
    result_iter = pool.imap_unordered(process_fn, row_generator(), chunksize=prefetch_batches)
    
    write_threshold = 50000
    
    # Stream results as they complete
    for chunk_rows in result_iter:
        pbar.update(len(chunk_rows))
        out_rows.extend(chunk_rows)
        
        # Write asynchronously when we have enough chunks
        if len(out_rows) >= write_threshold:
            tbl = rows_to_table(out_rows)
            writer.write(tbl)
            written += len(out_rows)
            out_rows.clear()
    
    # Write final batch
    if out_rows:
        tbl = rows_to_table(out_rows)
        writer.write(tbl)
        written += len(out_rows)
        out_rows.clear()
    
    pool.close()
    pool.join()
    
    # Wait for all writes to complete
    writer.close()
    
    pbar.close()
    print(f"[complete] total chunks written: {written}")

if __name__ == "__main__":
    run("/scratch/v13-ia-lake/data/")