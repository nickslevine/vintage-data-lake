from dataclasses import dataclass
import pathlib
from vintage_data_lake.utils import get_embeddings_ds
import random
import pyarrow.dataset as ds
import pyarrow as pa
import faiss
import numpy as np
import hashlib
from tqdm import tqdm
import time
import os


@dataclass
class IVFPQConfig:
    n_lists: int
    m: int
    n_bits: int
    batch_size: int
    vector_dim: int = 1024


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

def hash64(s: str) -> np.uint64:
    """
    Stable 64-bit ID from chunk_id (or any string).
    """
    h = hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest()
    return np.frombuffer(h, dtype=np.uint64)[0]

def table_ids_from_chunk_ids(table: pa.Table) -> np.ndarray:
    chunk_ids = table["chunk_id"].to_pylist()
    ids = np.fromiter((hash64(s) for s in chunk_ids), dtype=np.uint64, count=len(chunk_ids))
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

def run(config: IVFPQConfig, output_folder: pathlib.Path):
    # Set FAISS to use all available CPU cores
    n_threads = os.cpu_count() or 1
    faiss.omp_set_num_threads(n_threads)
    print(f"FAISS using {n_threads} threads")
    
    embeddings_ds = get_embeddings_ds()
    # cols = ["chunk_id", "doc_id", "vector", "year", "source"]
    print("Subsampling dataset...")
    # Increase training data from 1% to 10% for better CPU/memory utilization
    training_data = subsample_dataset_by_fragment(embeddings_ds, 0.05)
    print("Converting table to numpy...")
    X = table_to_numpy(training_data, "vector")
    print("Numpy array shape: %s" % (X.shape,))

    quantizer = faiss.IndexFlatL2(config.vector_dim)
    index = faiss.IndexIVFPQ(quantizer, config.vector_dim, config.n_lists, config.m, config.n_bits)
    index.verbose = True
    
    t0 = time.time()
    print("Training index on CPU...")
    index.train(X) # type: ignore
    print("Index trained in %.2f seconds" % (time.time() - t0))

    cols = ["chunk_id","doc_id", "year", "source", "vector"]

    scanner = embeddings_ds.scanner(columns=cols, batch_size=config.batch_size)
    total_rows = scanner.count_rows()

    pbar = tqdm(total=total_rows, desc="Adding vectors to index")
    meta_tables = []

    added_total = 0

    # Build on CPU for stability - GPU IVFPQ has issues with add_with_ids
    print("Adding vectors to CPU index...")
    for batch in scanner.to_batches():
        batch_table = pa.Table.from_batches([batch])
        X = table_to_numpy(batch_table, "vector")
        ids = table_ids_from_chunk_ids(batch_table).astype(np.int64, copy=False)  
        index.add_with_ids(X, ids) # type: ignore
        meta_tables.append(build_meta_table(ids, batch_table))
        N = len(batch_table)
        added_total += N
        pbar.update(N)

    pbar.close()

    print("Concatenating meta tables")
    meta_table: pa.Table = pa.concat_tables(meta_tables)

    print("Writing index")
    faiss.write_index(index, str(output_folder / "index.faiss"))
    print("Writing meta table")
    ds.write_dataset(
        data=meta_table,
        base_dir=output_folder / "meta_table",
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("year", pa.int32()), ("source", pa.string())]), flavor="hive"),
        existing_data_behavior="overwrite_or_ignore",
    )

if __name__ == "__main__":
    # Increase batch_size for better throughput and memory utilization
    config = IVFPQConfig(n_lists=2048, m=16, n_bits=8, batch_size=500000)
    path = pathlib.Path("/scratch/v13-ia-lake/faiss")
    run(config, path)