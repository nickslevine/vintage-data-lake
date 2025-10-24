from vintage_data_lake.chunk_docs import read_blob
import polars as pl
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class LakeConfig:
    path_documents: str = "/scratch/v13-ia-lake/data/parquet/documents"
    path_chunks: str = "/scratch/v13-ia-lake/data/parquet/chunks"
    path_blobs: str = "/scratch/v13-ia-lake/data/blobs"
    path_embeddings: str = "/scratch/v13-ia-lake/data/parquet/embeddings"
    overlap: int = 64
    tokenizer_model_name: str = "intfloat/e5-large-v2"


class Lake:
    def __init__(self, config: LakeConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model_name, use_fast=True)

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

    def get_chunks_from_doc_id(self, doc_id: str, select_columns = ["doc_id", "chunk_id", "seq", "text", "year", "source", "tokens"]) -> pl.DataFrame:
        return self.chunks.select(select_columns).filter(pl.col("doc_id") == doc_id).collect(engine="streaming")

    def get_chunk_from_chunk_id(self, chunk_id: str, select_columns = ["doc_id", "chunk_id", "seq", "text", "year", "source"]) -> pl.DataFrame:
        return self.chunks.select(select_columns).filter(pl.col("chunk_id") == chunk_id).collect(engine="streaming")

    def get_chunk_context(self, chunk_id: str, chunks_before: int = 1, chunks_after: int = 1):
        chunk = self.get_chunk_from_chunk_id(chunk_id, select_columns = ["seq", "doc_id", "chunk_id"])
        seq = chunk["seq"][0]
        doc_id = chunk["doc_id"][0]

        first_chunk = max(0, seq - chunks_before)
        last_chunk = seq + chunks_after

        doc_chunks = self.get_chunks_from_doc_id(doc_id)

        context_chunks = doc_chunks.filter(
            (pl.col("seq") >= first_chunk) & 
            (pl.col("seq") <= last_chunk)
        )
        return context_chunks

    def merge_sequences(self, sequences: list[list[int]]) -> str:
        """Expects a list of lists of tokens (eg, from the 'tokens' column after calling get_chunk_context)"""
        def _strip_token_seq(tokens):
            return tokens[1:-1]
        merged = _strip_token_seq(sequences[0])
        for seq in sequences[1:]:
            stripped = _strip_token_seq(seq)
            assert merged[-self.config.overlap:] == stripped[:self.config.overlap]
            merged += stripped[self.config.overlap:]
        return self.tokenizer.decode(merged)

    def get_documents_by_date_range(self, year_min: int, year_max: int) -> pl.DataFrame:
        cols = ['doc_id', 'title', 'author', 'text_uri', 'preview', 'year', 'source']
        return self.documents.select(cols).filter((pl.col("year")<=year_max) & (pl.col("year")>=year_min)).collect(engine="streaming")