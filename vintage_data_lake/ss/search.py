import faiss
import pathlib
import polars as pl
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


def load_index(path: pathlib.Path, n_probe: int):
    """
    Load FAISS IVFPQ index from disk (CPU-based for stability).
    
    Args:
        path: Path to the .faiss index file
        n_probe: Number of clusters to probe during search
        
    Returns:
        IVFPQIndex with CPU index
    """
    import os
    
    index = faiss.read_index(str(path))
    index.nprobe = n_probe
    
    # Enable multi-threading for faster CPU search
    n_threads = os.cpu_count() or 1
    faiss.omp_set_num_threads(n_threads)
    
    return index


def load_meta_table(path: pathlib.Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def embed_query(query: str, model_name: str = "intfloat/e5-large-v2") -> np.ndarray:

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    
    # Prefix query (e5 models require "query:" prefix for queries)
    prefixed_query = f"query: {query}"
    
    # Tokenize with padding and truncation to match training
    inputs = tokenizer(
        prefixed_query,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling with attention mask (same as in chunks_to_embeddings.py)
    attention_mask = inputs["attention_mask"]
    mask = attention_mask.unsqueeze(-1).type_as(outputs.last_hidden_state)
    summed = (outputs.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    
    # L2 normalize (CRITICAL - index embeddings are normalized!)
    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
    
    return normalized.detach().cpu().numpy().astype(np.float32)

def search_with_meta(index: faiss.Index, meta_table: pl.LazyFrame, query: str, n_results: int = 10) -> pl.DataFrame:
    query_embedding = embed_query(query)
    distances, labels = index.search(query_embedding, n_results) # type: ignore
    search_results = pl.DataFrame({"distances": distances.squeeze(), "labels": labels.squeeze()})
    mt_filtered = meta_table.filter(pl.col("faiss_id").is_in(search_results["labels"].to_list())).collect()
    return search_results.join(mt_filtered, left_on="labels", right_on="faiss_id", how="left")


def get_filter_ids(year_min: int, year_max: int) -> np.ndarray:
    mt = load_meta_table(pathlib.Path("/scratch/v13-ia-lake/faiss/meta_table"))
    selected_ids = mt.filter(pl.col("year").is_in(range(year_min, year_max + 1))).select("faiss_id").collect(engine="streaming")["faiss_id"].to_numpy().astype(np.int64, copy=False)
    return selected_ids

def search_with_filter(index: faiss.Index, query: str, n_results: int = 10, year_min: int = 1400, year_max: int = 1930) -> pl.DataFrame:
    selected_ids = get_filter_ids(year_min, year_max)
    params = faiss.SearchParametersIVF()
    params.nprobe = 64
    params.sel = faiss.IDSelectorBatch(selected_ids) # type: ignore
    query_embedding = embed_query(query)
    distances, labels = index.search(query_embedding, n_results, params=params) # type: ignore
    search_results = pl.DataFrame({"distances": distances.squeeze(), "labels": labels.squeeze()})
    meta_table = load_meta_table(pathlib.Path("/scratch/v13-ia-lake/faiss/meta_table"))
    mt_filtered = meta_table.filter(pl.col("faiss_id").is_in(search_results["labels"].to_list())).collect()
    return search_results.join(mt_filtered, left_on="labels", right_on="faiss_id", how="left")


def add_chunks_to_search_results(search_results: pl.DataFrame) -> pl.DataFrame:
    chunks_df = pl.scan_parquet("/scratch/v13-ia-lake/data/parquet/chunks")
    joined = chunks_df.join(search_results.lazy(), on=["chunk_id", "year", "source"], how="semi").collect(engine="streaming") # type: ignore
    return joined

def search_with_text(idx: faiss.Index, query: str, n_results: int = 10) -> pl.DataFrame:
    mt = load_meta_table(pathlib.Path("/scratch/v13-ia-lake/faiss/meta_table"))
    results = search_with_meta(idx, mt, query, n_results)
    with_chunks = add_chunks_to_search_results(results)
    return with_chunks