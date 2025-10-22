import torch
from transformers import AutoModel, AutoTokenizer
import concurrent.futures
from math import ceil
import queue
import threading
from functools import partial

import pyarrow as pa
import pyarrow.dataset as ds
from datetime import datetime
from datetime import timezone
import numpy as np
import os
from tqdm import tqdm

def make_multi_gpu_encoders(
    model_name: str = "intfloat/e5-large-v2",
    device_ids: list[int] | None = None,
    dtype: str = "bfloat16",
    max_length: int = 512,
):
    """
    Create GPU encoder functions that accept PRE-TOKENIZED inputs.
    This avoids GIL contention - tokenize once, then shard tokens to GPUs.
    
    Returns:
        (tokenizer, list of encode functions)
    """
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    # dtype mapping
    if dtype.lower() in ("bf16", "bfloat16"):
        torch_dtype = torch.bfloat16
    elif dtype.lower() in ("fp16", "float16", "half"):
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Shared tokenizer (used once per batch, not per GPU)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    encode_fns = []

    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    for dev in device_ids:
        device = f"cuda:{dev}"
        mdl = AutoModel.from_pretrained(model_name, dtype=torch_dtype).to(device).eval()

        # Create encode function that accepts PRE-TOKENIZED input
        def make_encode_fn(_mdl=mdl, _device=device):
            def encode_fn(input_ids, attention_mask, batch_size: int = 512):
                """
                Encode pre-tokenized inputs.
                
                Args:
                    input_ids: tensor [N, seq_len]
                    attention_mask: tensor [N, seq_len]
                    batch_size: chunk size for processing
                """
                torch.set_grad_enabled(False)
                
                N = input_ids.shape[0]
                if N == 0:
                    return torch.empty(0, _mdl.config.hidden_size)
                
                out_chunks = []
                for i in range(0, N, batch_size):
                    batch_input_ids = input_ids[i:i + batch_size].to(_device, non_blocking=True)
                    batch_attention_mask = attention_mask[i:i + batch_size].to(_device, non_blocking=True)
                    
                    with torch.inference_mode():
                        out = _mdl(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                        pooled = _mean_pool(out.last_hidden_state, batch_attention_mask)
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    
                    # Use pinned memory for faster CPU transfer
                    out_cpu = torch.empty(
                        pooled.shape, 
                        dtype=torch.float32, 
                        pin_memory=True
                    )
                    out_cpu.copy_(pooled, non_blocking=True)
                    out_chunks.append(out_cpu)
                
                # Ensure all async transfers complete
                torch.cuda.current_stream(_device).synchronize()
                return torch.cat(out_chunks, dim=0) if out_chunks else torch.empty(0, _mdl.config.hidden_size)
            return encode_fn

        encode_fns.append(make_encode_fn())

    return tokenizer, encode_fns

def embed_with_replicas(
    input_ids: torch.Tensor,      # Pre-tokenized [N, seq_len]
    attention_mask: torch.Tensor,  # Pre-tokenized [N, seq_len]
    encode_fns: list,              # GPU encode functions
    batch_size: int = 512,
    shard: str = "contiguous",     # or "round_robin"
):
    """
    Shard pre-tokenized inputs to GPUs for parallel inference.
    NO tokenization happens here - it's done in the prefetch thread.
    
    Args:
        input_ids: Pre-tokenized input IDs [N, seq_len]
        attention_mask: Pre-tokenized attention mask [N, seq_len]
        encode_fns: List of GPU encode functions (one per device)
        batch_size: Batch size for GPU processing
        shard: How to distribute work ("contiguous" or "round_robin")
    """
    N = input_ids.shape[0]
    G = len(encode_fns)
    if N == 0:
        return torch.empty(0, 0)

    # Build shard assignments (shard the pre-tokenized tensors)
    assignments = []
    if shard == "contiguous":
        per = ceil(N / G)
        for g in range(G):
            start = g * per
            end = min(N, (g + 1) * per)
            if start >= end:
                assignments.append(([], None, None))
            else:
                idxs = list(range(start, end))
                assignments.append((idxs, input_ids[start:end], attention_mask[start:end]))
    elif shard == "round_robin":
        bucket_idxs = [[] for _ in range(G)]
        for i in range(N):
            g = i % G
            bucket_idxs[g].append(i)
        
        for g in range(G):
            if not bucket_idxs[g]:
                assignments.append(([], None, None))
            else:
                idxs = bucket_idxs[g]
                assignments.append((
                    idxs,
                    input_ids[idxs],
                    attention_mask[idxs]
                ))
    else:
        raise ValueError("shard must be 'contiguous' or 'round_robin'")

    # Worker: PURE GPU INFERENCE (no tokenization, no GIL contention!)
    def _worker(gpu_i: int, idxs: list[int], shard_input_ids, shard_attention_mask):
        if not idxs or shard_input_ids is None:
            return gpu_i, idxs, None
        # GPU work releases GIL - all 8 GPUs run in true parallel
        embs = encode_fns[gpu_i](shard_input_ids, shard_attention_mask, batch_size=batch_size)
        return gpu_i, idxs, embs

    # Launch all GPUs in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=G) as ex:
        futs = [
            ex.submit(_worker, gpu_i, idxs, shard_ids, shard_mask)
            for gpu_i, (idxs, shard_ids, shard_mask) in enumerate(assignments)
        ]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    # Determine embedding dim from first non-empty result
    emb_dim = None
    for _, idxs, embs in results:
        if embs is not None and embs.numel() > 0:
            emb_dim = embs.shape[1]
            break
    if emb_dim is None:
        return torch.empty(0, 0)

    # Stitch back into original order
    out = torch.empty(N, emb_dim, dtype=results[0][2].dtype if results[0][2] is not None else torch.float32)
    for _, idxs, embs in results:
        if not idxs:
            continue
        out[idxs] = embs
    return out


def _tokenize_chunk(texts_chunk, model_name, max_length):
    """Worker function for multi-process tokenization."""
    # Each process loads its own tokenizer (can't pickle HF tokenizers)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    prefixed = [f"passage: {t}" for t in texts_chunk]
    encoded = tokenizer(
        prefixed,
        padding='max_length',  # Pad to max_length so all chunks have same shape
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


class MultiProcessPrefetchLoader:
    """
    Async batch loader with MULTI-PROCESS tokenization.
    Uses N CPU cores to tokenize in parallel, keeping up with fast GPUs.
    """
    def __init__(
        self, 
        scanner, 
        model_name: str,
        max_length: int = 512, 
        prefetch_size: int = 4,
        num_tokenizer_workers: int = 4,
    ):
        """
        Args:
            scanner: PyArrow dataset scanner
            model_name: Model name for tokenizer
            max_length: Max sequence length for tokenization
            prefetch_size: Number of batches to prefetch ahead
            num_tokenizer_workers: Number of CPU cores for parallel tokenization
        """
        self.scanner = scanner
        self.model_name = model_name
        self.max_length = max_length
        self.prefetch_size = prefetch_size
        self.num_tokenizer_workers = num_tokenizer_workers
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.thread = None
        self.stop_event = threading.Event()
        self.process_pool = None
        
    def _loader_worker(self):
        """Background thread that loads and coordinates multi-process tokenization."""
        try:
            # Create process pool for tokenization
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_tokenizer_workers
            ) as pool:
                for batch in self.scanner.to_batches():
                    if self.stop_event.is_set():
                        break
                    
                    # Load and convert Arrow batch
                    batch_table = pa.Table.from_batches([batch])
                    texts = batch_table["text"].to_pylist()
                    
                    # PARALLEL TOKENIZATION across N processes
                    # Split texts into chunks for parallel processing
                    chunk_size = max(1, len(texts) // self.num_tokenizer_workers)
                    text_chunks = [
                        texts[i:i+chunk_size] 
                        for i in range(0, len(texts), chunk_size)
                    ]
                    
                    # Tokenize chunks in parallel
                    tokenize_fn = partial(
                        _tokenize_chunk,
                        model_name=self.model_name,
                        max_length=self.max_length
                    )
                    futures = [pool.submit(tokenize_fn, chunk) for chunk in text_chunks]
                    
                    # Gather results
                    input_id_chunks = []
                    attention_mask_chunks = []
                    for fut in futures:
                        input_ids, attention_mask = fut.result()
                        input_id_chunks.append(input_ids)
                        attention_mask_chunks.append(attention_mask)
                    
                    # Concatenate all chunks
                    if input_id_chunks:
                        all_input_ids = torch.cat(input_id_chunks, dim=0)
                        all_attention_masks = torch.cat(attention_mask_chunks, dim=0)
                    else:
                        all_input_ids = torch.empty(0, 0, dtype=torch.long)
                        all_attention_masks = torch.empty(0, 0, dtype=torch.long)
                    
                    # Queue pre-tokenized batch
                    self.queue.put((
                        batch_table,
                        all_input_ids,
                        all_attention_masks
                    ))
                
                # Signal completion
                self.queue.put(None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.queue.put(e)
    
    def start(self):
        """Start the prefetch thread (which manages process pool)."""
        self.thread = threading.Thread(target=self._loader_worker, daemon=True)
        self.thread.start()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item
    
    def stop(self):
        """Stop the prefetch thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)


class AsyncWriter:
    """
    Async writer that writes embeddings to disk in a background thread.
    Prevents disk I/O from blocking GPU work.
    """
    def __init__(self, base_dir: str, model_name: str, queue_size: int = 3):
        """
        Args:
            base_dir: Base directory for embeddings
            model_name: Model name for partitioning
            queue_size: Max number of pending writes
        """
        self.base_dir = base_dir
        self.model_name = model_name
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = None
        self.stop_event = threading.Event()
        
    def _writer_worker(self):
        """Background thread that writes to disk."""
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                if item is None:  # Shutdown signal
                    break
                    
                chunk_table, embeddings, write_i, run_id = item
                write_embeddings_dataset(
                    base_dir=self.base_dir,
                    model_name=self.model_name,
                    chunk_table=chunk_table,
                    embeddings=embeddings,
                    write_i=write_i,
                    run_id=run_id,
                )
                self.queue.task_done()
        except Exception as e:
            print(f"Writer thread error: {e}")
    
    def start(self):
        """Start the writer thread."""
        self.thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.thread.start()
    
    def submit(self, chunk_table: pa.Table, embeddings, write_i: int, run_id: str | None = None):
        """Submit a write job (blocks if queue is full)."""
        self.queue.put((chunk_table, embeddings, write_i, run_id))
    
    def shutdown(self, wait: bool = True):
        """Shutdown the writer thread."""
        self.queue.put(None)  # Shutdown signal
        if wait and self.thread:
            self.queue.join()  # Wait for all writes to complete
            self.thread.join(timeout=30)


def write_embeddings_dataset(
    base_dir: str,
    model_name: str,
    chunk_table: pa.Table,
    write_i: int,
    embeddings: "torch.Tensor | np.ndarray",
    run_id: str | None = None,
):
    """
    Write chunk-level embeddings to a Parquet dataset partitioned by
    model / year / source.

    Args:
        base_dir: base path, e.g. "/scratch/v13-ia-lake/data"
        model_name: short model name, e.g. "e5-large-v2"
        chunk_table: Arrow Table with at least columns
                     ["chunk_id", "doc_id", "year", "source"]
        embeddings: torch.Tensor [N, D] or np.ndarray [N, D]
        run_id: optional unique string for file naming
    """

    if isinstance(embeddings, torch.Tensor):
        embs = embeddings.cpu().numpy()
    else:
        embs = embeddings
    N, D = embs.shape

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    # Normalize column alignment
    cols = ["chunk_id", "doc_id", "year", "source"]
    meta_tbl = chunk_table.select(cols)
    assert len(meta_tbl) == N, "Mismatch between chunk rows and embeddings"

    # Build Arrow arrays
    vec_array = pa.FixedSizeListArray.from_arrays(pa.array(embs.ravel(), pa.float32()), D)
    model_col = pa.array([model_name] * N, pa.string())
    created_at = pa.array([datetime.now( timezone.utc)] * N, pa.timestamp("us"))

    out_tbl = meta_tbl.append_column("model", model_col)
    out_tbl = out_tbl.append_column("vector", vec_array)
    out_tbl = out_tbl.append_column("dim",  pa.array([D] * N, type=pa.int32()))
    out_tbl = out_tbl.append_column("created_at", created_at)

    # Write to dataset
    out_path = os.path.join(base_dir, "parquet", "embeddings", model_name)
    schema = pa.schema([
        ("year", pa.int32()),
        ("source", pa.string())
    ])
    ds.write_dataset(
        data=out_tbl,
        base_dir=out_path,
        format="parquet",
        partitioning=ds.partitioning(schema, flavor="hive"),
        existing_data_behavior="overwrite_or_ignore",
        basename_template=f"part-{run_id}-{write_i}-{{i}}.parquet",
    )
    return len(out_tbl)

def run_pipelined_embedding(
    dataset_path: str,
    output_base_dir: str,
    model_name: str = "e5-large-v2",
    batch_size: int = 64000,
    encode_batch_size: int = 1024,
    prefetch_size: int = 4,
    write_queue_size: int = 3,
    num_tokenizer_workers: int = 8,
    device_ids: list[int] | None = None,
):
    """
    Run embedding generation with full pipeline parallelism + multi-process tokenization.
    
    Pipeline stages run concurrently:
    - Stage 1 (N processes): Parallel tokenization on CPU
    - Stage 2 (main thread): Shard tokens to 8 GPUs
    - Stage 3 (8 GPU threads): Parallel inference
    - Stage 4 (background thread): Write embeddings to disk
    
    Args:
        dataset_path: Path to chunks dataset
        output_base_dir: Base directory for output embeddings
        model_name: Model name/path
        batch_size: Number of rows per Arrow batch
        encode_batch_size: Batch size for GPU encoding
        prefetch_size: Number of batches to prefetch (increase if GPUs are fast)
        write_queue_size: Max pending writes (2-3 to avoid memory bloat)
        num_tokenizer_workers: CPU cores for parallel tokenization (match GPU count)
        device_ids: GPU device IDs to use (None = all available)
    """
    print("Making encoders...")
    tokenizer, encode_fns = make_multi_gpu_encoders(
        model_name=model_name,
        device_ids=device_ids,
    )
    print(f"Encoders made for {len(encode_fns)} GPUs.")
    
    print("Loading dataset...")
    chunks = ds.dataset(dataset_path, format="parquet", partitioning="hive")
    scanner = chunks.scanner(
        batch_size=batch_size,
        columns=["chunk_id", "doc_id", "year", "source", "text"]
    )
    total_row_count = scanner.count_rows()
    print(f"Dataset loaded. Total rows: {total_row_count:,}")
    
    # Start async writer
    writer = AsyncWriter(
        base_dir=output_base_dir,
        model_name=model_name.split("/")[-1],  # Use short name
        queue_size=write_queue_size,
    )
    writer.start()
    
    # Start multi-process prefetch loader (parallel tokenization)
    loader = MultiProcessPrefetchLoader(
        scanner,
        model_name=model_name,
        max_length=512,
        prefetch_size=prefetch_size,
        num_tokenizer_workers=num_tokenizer_workers,
    )
    loader.start()
    
    # Main pipeline loop
    write_i = 0
    pbar = tqdm(desc="Generating embeddings", total=total_row_count, unit=" chunks")
    
    try:
        for batch_table, input_ids, attention_mask in loader:
            # GPU encoding: inputs are PRE-TOKENIZED, just pure GPU inference
            embs = embed_with_replicas(
                input_ids,
                attention_mask,
                encode_fns,
                batch_size=encode_batch_size
            )
            
            # Submit to async writer (non-blocking unless queue is full)
            writer.submit(
                chunk_table=batch_table,
                embeddings=embs,
                write_i=write_i,
            )
            
            pbar.update(len(batch_table))
            write_i += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down gracefully...")
        loader.stop()
        writer.shutdown(wait=True)
        pbar.close()
        raise
    
    # Clean shutdown
    pbar.close()
    loader.stop()
    writer.shutdown(wait=True)
    print("Pipeline complete!")


if __name__ == "__main__":
    run_pipelined_embedding(
        dataset_path="/scratch/v13-ia-lake/data/parquet/chunks",
        output_base_dir="/scratch/v13-ia-lake/data",
        model_name="intfloat/e5-large-v2",
        batch_size=64000,
        encode_batch_size=2048,
        prefetch_size=4,              # Increase since tokenization is now faster
        write_queue_size=3,
        num_tokenizer_workers=8,      # Match number of GPUs for balanced pipeline
    )