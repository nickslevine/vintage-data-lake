import pandas as pd
from vintage_data_lake.chunk_docs import read_blob
import polars as pl
from dataclasses import dataclass





# def get_doc_metadata():
#     return pd.read_parquet("/scratch/v13-ia-lake/data/parquet/documents")

# def get_embeddings_ds():
#     return ds.dataset("/scratch/v13-ia-lake/data/parquet/embeddings/e5-large-v2", format="parquet", partitioning="hive")

# def get_embeddings_pl():
#     return pl.scan_parquet("/scratch/v13-ia-lake/data/parquet/embeddings/e5-large-v2")


# def get_document_from_doc_id(doc_metadata: pd.DataFrame, doc_id: str):
#     text_uri = doc_metadata[doc_metadata.doc_id == doc_id].text_uri.values[0]
#     return read_blob(text_uri, "/scratch/v13-ia-lake/data/blobs")

@dataclass
class LakeConfig:
    path_documents: str = "/scratch/v13-ia-lake/data/parquet/documents"
    path_chunks: str = "/scratch/v13-ia-lake/data/parquet/chunks"
    path_blobs: str = "/scratch/v13-ia-lake/data/blobs"
    path_embeddings: str = "/scratch/v13-ia-lake/data/parquet/embeddings"


class Lake:
    def __init__(self, config: LakeConfig):
        self.config = config

    @property
    def documents(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.config.path_documents)
    
    @property
    def chunks(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.config.path_chunks)
    
    @property
    def embeddings(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.config.path_embeddings)

    def get_document_from_doc_id(self, doc_id: str) -> str:
        uri =  self.documents.filter(pl.col("doc_id") == doc_id).collect(engine="streaming")["text_uri"][0]
        return read_blob(uri, self.config.path_blobs)

    def get_chunks_from_doc_id(self, doc_id: str, select_columns = ["doc_id", "chunk_id", "seq", "text", "year", "source"]) -> pl.DataFrame:
        return self.chunks.select(select_columns).filter(pl.col("doc_id") == doc_id).collect(engine="streaming")

    def get_chunk_from_chunk_id(self, chunk_id: str, select_columns = ["doc_id", "chunk_id", "seq", "text", "year", "source"]) -> pl.DataFrame:
        return self.chunks.select(select_columns).filter(pl.col("chunk_id") == chunk_id).collect(engine="streaming")

    def get_chunk_context(self, chunk_id: str, chunks_before: int = 1, chunks_after: int = 1):
        chunk = self.get_chunk_from_chunk_id(chunk_id, select_columns = ["seq", "doc_id", "chunk_id"])
        seq = chunk["seq"][0]
        doc_id = chunk["doc_id"][0]
        print(seq, doc_id)

        first_chunk = max(0, seq - chunks_before)
        last_chunk = seq + chunks_after

        doc_chunks = self.get_chunks_from_doc_id(doc_id)
        print("got doc chunks")

        context_chunks = doc_chunks.filter(
            (pl.col("seq") >= first_chunk) & 
            (pl.col("seq") <= last_chunk)
        )
        return context_chunks