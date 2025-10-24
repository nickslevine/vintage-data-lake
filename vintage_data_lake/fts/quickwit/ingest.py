import json
import os
import time
from typing import Iterable, Dict
import pyarrow.dataset as ds

import requests
import pyarrow as pa
QW_ENDPOINT = os.environ.get("QW_ENDPOINT", "http://127.0.0.1:7280")
INDEX_ID = os.environ.get("QW_INDEX", "chunks")

def rows_from_table(tbl: pa.Table) -> Dict[str, list]:
    cols_needed = ["chunk_id", "doc_id", "year", "source", "text"]
    cols = {}
    for c in cols_needed:
        if c not in tbl.column_names:
            raise ValueError(f"Missing required column `{c}` in batch: {tbl.column_names}")
        cols[c] = tbl[c].to_pylist()
    return cols
    

def ndjson_iter(tbl: pa.Table) -> Iterable[str]:
    cols = rows_from_table(tbl)
    N = len(tbl)
    texts = cols["text"]
    for i in range(N):
        doc = {
            "chunk_id": cols["chunk_id"][i],
            "doc_id":   cols["doc_id"][i],
            "year":     int(cols["year"][i]),
            "source":   cols["source"][i],
            "text":     texts[i],
        }

        yield json.dumps(doc, ensure_ascii=False) + "\n"


def post_ndjson(lines: Iterable[str],
                endpoint: str,
                timeout: int = 120,
                gzip: bool = False) -> requests.Response:
    url = f"{endpoint}/api/v1/{INDEX_ID}/ingest"
    headers = {"Content-Type": "application/x-ndjson"}
    data = "".join(lines)
    if gzip:
        import gzip as gz
        data = gz.compress(data.encode("utf-8"))
        headers["Content-Encoding"] = "gzip"
    r = requests.post(url, data=data, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


def ingest_batches(batch_iter: Iterable[pa.Table],
                   batch_docs_cap: int = 2000,
                   gzip: bool = False,
                   ) -> None:
    """
    batch_iter: yields pyarrow.Table batches with required columns
    batch_docs_cap: hard cap to split very large Arrow batches
    watermark_path: optional file path where we write the last successful ingest counter
    """
    total_docs = 0
    batch_i = 0

    for tbl in batch_iter:
        # Some scanners give big batches; optionally split into smaller slices
        size = len(tbl)
        start = 0
        while start < size:
            end = min(start + batch_docs_cap, size)
            slice_tbl = tbl.slice(start, end - start)

            lines = list(ndjson_iter(slice_tbl))
            # Backoff+retry loop
            for attempt in range(5):
                try:
                    _ = post_ndjson(lines, QW_ENDPOINT, timeout=120, gzip=gzip)
                    break
                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    print(f"[WARN] ingest attempt {attempt+1} failed: {e}; retrying in {wait}s...")
                    time.sleep(wait)
            else:
                raise RuntimeError("Ingest failed after retries")

            total_docs += len(lines)
            batch_i += 1

            print(f"[OK] committed batch {batch_i} (+{len(lines)} docs) total={total_docs}")

            start = end

    print(f"[DONE] Ingest complete. total_docs={total_docs}")

def iterator_from_parquet(chunks_parquet_dir: str,
                               rows_per_batch: int = 50_000) -> Iterable[pa.Table]:

    dataset = ds.dataset(chunks_parquet_dir, format="parquet", partitioning="hive")
    scanner = dataset.scanner(batch_size=rows_per_batch, columns=["chunk_id","doc_id","year","source","text"])
    for rb in scanner.to_batches():
        yield pa.Table.from_batches([rb])


if __name__ == "__main__":
    iterator = iterator_from_parquet("/scratch/v13-ia-lake/data/parquet/chunks")
    ingest_batches(iterator)