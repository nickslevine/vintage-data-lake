import pandas as pd
from vintage_data_lake.chunk_docs import read_blob
import pyarrow.dataset as ds
import polars as pl





def get_doc_metadata():
    return pd.read_parquet("/scratch/v13-ia-lake/data/parquet/documents")

def get_embeddings_ds():
    return ds.dataset("/scratch/v13-ia-lake/data/parquet/embeddings/e5-large-v2", format="parquet", partitioning="hive")

def get_embeddings_pl():
    return pl.scan_parquet("/scratch/v13-ia-lake/data/parquet/embeddings/e5-large-v2")


def get_document_from_doc_id(doc_metadata: pd.DataFrame, doc_id: str):
    text_uri = doc_metadata[doc_metadata.doc_id == doc_id].text_uri.values[0]
    return read_blob(text_uri, "/scratch/v13-ia-lake/data/blobs")