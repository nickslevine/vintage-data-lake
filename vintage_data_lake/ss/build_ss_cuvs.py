import cupy as cp
from cuvs.common import Resources 
from cuvs.neighbors import ivf_pq, ivf_flat
from dataclasses import dataclass
import pathlib
import random
import pyarrow.dataset as ds
import pyarrow as pa
import faiss
import numpy as np
import hashlib
from tqdm import tqdm
import time
import os




def get_embeddings_ds():
    return ds.dataset("/scratch/v13-ia-lake/data/parquet/embeddings/e5-large-v2", format="parquet", partitioning="hive")

def subsample_dataset_by_fragment(dataset: ds.Dataset, fraction: float = 0.1) -> pa.Table:
    random.seed(42)
    fragments = list(dataset.get_fragments())
    print(f"Sampling from { int(len(fragments) * fraction)} fragments out of {len(fragments)}")
    sampled = random.sample(fragments, int(len(fragments) * fraction))
    tables = [fragment.to_table() for fragment in sampled]
    return pa.concat_tables(tables)

def table_to_numpy(table: pa.Table, column_name: str = "vector") -> np.ndarray:
    column = table[column_name]
    column_combined = column.combine_chunks()
    values = column_combined.values.to_numpy()
    reshaped = values.reshape(len(column_combined), -1)
    return np.ascontiguousarray(reshaped, dtype=np.float32)

def hash64(s: str) -> np.int64:
    """
    Stable 64-bit ID from chunk_id (or any string).
    """
    h = hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest()
    return np.frombuffer(h, dtype=np.int64)[0]

def table_ids_from_chunk_ids(table: pa.Table) -> np.ndarray:
    chunk_ids = table["chunk_id"].to_pylist()
    ids = np.fromiter((hash64(s) for s in chunk_ids), dtype=np.int64, count=len(chunk_ids))
    return ids

def build_meta_table(ids: np.ndarray, tbl: pa.Table) -> pa.Table:

    meta_tbl = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),  # Faiss uses int64 with INDICES_64_BIT
            tbl["chunk_id"],
            tbl["doc_id"],
            tbl["year"].cast(pa.int32()) if tbl["year"].type != pa.int32() else tbl["year"],
            tbl["source"],
        ],
        names=["faiss_id", "chunk_id", "doc_id", "year", "source"],
    )
    return meta_tbl


if __name__ == "__main__":
    
    embeddings_ds = get_embeddings_ds()
    # cols = ["chunk_id", "doc_id", "vector", "year", "source"]
    print("Subsampling dataset...")
    # Increase training data from 1% to 10% for better CPU/memory utilization
    training_data = subsample_dataset_by_fragment(embeddings_ds, 0.1)
    print("Converting table to numpy...")
    X = table_to_numpy(training_data, "vector")

    train_device = cp.asarray(X, dtype=cp.float32)
    print("Train device shape:", train_device.shape)

    # Build index parameters
    build_params = ivf_pq.IndexParams(
        metric="sqeuclidean",
        n_lists=65536,
        pq_dim=64,
        pq_bits=8,
        kmeans_trainset_fraction=1.0, 
        add_data_on_build=False  
    )

    print("Building index...")
    t0 = time.time()
    index = ivf_pq.build(build_params, train_device)
    cp.cuda.Stream.null.synchronize()
    print("Index built in %.2f seconds" % (time.time() - t0))

    cols = ["chunk_id","doc_id", "year", "source", "vector"]
    scanner = embeddings_ds.scanner(columns=cols, batch_size=500000)
    total_rows = scanner.count_rows()

    pbar = tqdm(total=total_rows, desc="Adding vectors to index")
    meta_tables = []


    for batch in scanner.to_batches():
        batch_table = pa.Table.from_batches([batch])
        X = table_to_numpy(batch_table, "vector")
        ids = table_ids_from_chunk_ids(batch_table).astype(np.int64, copy=False)  
        ids_device = cp.asarray(ids, dtype=np.int64)
        batch_device = cp.asarray(X, dtype=cp.float32)
        index = ivf_pq.extend(index, batch_device, ids_device)
        cp.cuda.Stream.null.synchronize()
        meta_tables.append(build_meta_table(ids, batch_table))
        pbar.update(len(ids))
    pbar.close()

    print("Saving index...")
    ivf_pq.save("/scratch/v13-ia-lake/cuvs/ivf_pq.bin", index)
    meta_table: pa.Table = pa.concat_tables(meta_tables)
    print("Writing meta table...")
    ds.write_dataset(
        data=meta_table,
        base_dir="/scratch/v13-ia-lake/cuvs/meta_table",
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("year", pa.int32()), ("source", pa.string())]), flavor="hive"),
        existing_data_behavior="overwrite_or_ignore",
    )


