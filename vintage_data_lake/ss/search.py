import faiss
import pathlib
import polars as pl
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
from dataclasses import dataclass

@dataclass
class SemanticSearchConfig:
    embedding_model_name: str = "intfloat/e5-large-v2"
    index_path: str = "/scratch/v13-ia-lake/faiss/index.faiss"
    meta_table_path: str = "/scratch/v13-ia-lake/faiss/meta_table"
    chunks_path: str = "/scratch/v13-ia-lake/data/parquet/chunks"

class SemanticSearch:
    def __init__(self, config: SemanticSearchConfig):
        self.config = config
        self.index = self.load_index(config.index_path)
        self.meta_table = pl.scan_parquet(config.meta_table_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(config.embedding_model_name)
        self.chunks = pl.scan_parquet(config.chunks_path)

    def load_index(self, path: str):
        index = faiss.read_index(str(path))
        n_threads = os.cpu_count() or 1
        faiss.omp_set_num_threads(n_threads)        
        return index

    def embed_query(self, query: str) -> np.ndarray:

        prefixed_query = f"query: {query}"

        inputs = self.tokenizer(
            prefixed_query,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling with attention mask (same as in chunks_to_embeddings.py)
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).type_as(outputs.last_hidden_state)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        
        # L2 normalize (CRITICAL - index embeddings are normalized!)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        
        return normalized.detach().cpu().numpy().astype(np.float32)

    def get_filter_ids(self, year_min: int, year_max: int) -> np.ndarray:
        selected_ids = self.meta_table.filter(pl.col("year").is_in(range(year_min, year_max + 1))).select("faiss_id").collect(engine="streaming")["faiss_id"].to_numpy().astype(np.int64, copy=False)
        return selected_ids

    def add_chunks_to_search_results(self, search_results: pl.DataFrame) -> pl.DataFrame:
        joined = self.chunks.join(search_results.lazy(), on=["chunk_id", "year", "source"], how="semi").collect(engine="streaming") # type: ignore
        return joined

    def search(self,query, n=10, year_min = None, year_max = None, n_probe = 64, with_text = True) -> pl.DataFrame:
        params = faiss.SearchParametersIVF()
        params.nprobe = n_probe
        if year_min is not None or year_max is not None:
            if year_min is None:
                year_min = 0
            if year_max is None:
                year_max = 10000
            selected_ids = self.get_filter_ids(year_min, year_max)
            print(f"Filtering by years: {year_min} to {year_max}. Found {len(selected_ids)} ids.")
            params.sel = faiss.IDSelectorBatch(selected_ids) # type: ignore
        query_embedding = self.embed_query(query)
        distances, labels = self.index.search(query_embedding, n, params=params) # type: ignore
        search_results = pl.DataFrame({"distances": distances.squeeze(), "labels": labels.squeeze()})
        mt_filtered = self.meta_table.filter(pl.col("faiss_id").is_in(search_results["labels"].to_list())).collect()
        joined = search_results.join(mt_filtered, left_on="labels", right_on="faiss_id", how="left")
        if with_text:
            joined = self.add_chunks_to_search_results(joined)
        return joined