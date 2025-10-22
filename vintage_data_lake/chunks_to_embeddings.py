import torch
from transformers import AutoModel, AutoTokenizer
import concurrent.futures
from math import ceil

def make_multi_gpu_encoders(
    model_name: str = "intfloat/e5-large-v2",
    device_ids: list[int] | None = None,
    dtype: str = "bfloat16",
    max_length: int = 512,
):

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    # dtype mapping
    if dtype.lower() in ("bf16", "bfloat16"):
        torch_dtype = torch.bfloat16
    elif dtype.lower() in ("fp16", "float16", "half"):
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    encode_fns = []

    for dev in device_ids:
        device = f"cuda:{dev}"
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mdl = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(device).eval()

        def _mean_pool(last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts

        # bind locals so each closure keeps its own objects/device
        def make_encode_fn(_tok=tok, _mdl=mdl, _device=device, _max_len=max_length):
            def encode_fn(texts, batch_size: int = 512):
                torch.set_grad_enabled(False)

                if not isinstance(texts, (list, tuple)):
                    texts = [texts]
                prefixed = [f"passage: {t}" for t in texts]

                out_chunks = []
                for i in range(0, len(prefixed), batch_size):
                    batch = prefixed[i:i + batch_size]
                    enc = _tok(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=_max_len,
                        return_tensors="pt",
                    )
                    # IMPORTANT: move *inputs* to the *model's* device
                    enc = {k: v.to(_device, non_blocking=True) for k, v in enc.items()}
                    with torch.inference_mode():
                        out = _mdl(**enc)
                        pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    out_chunks.append(pooled.to(dtype=torch.float32).cpu())
                return torch.cat(out_chunks, dim=0) if out_chunks else torch.empty(0, _mdl.config.hidden_size)
            return encode_fn

        encode_fns.append(make_encode_fn())

    return encode_fns


def embed_with_replicas(
    texts: list[str],
    encode_fns: list,          # from make_multi_gpu_encoders(...)
    batch_size: int = 512,
    shard: str = "contiguous", # or "round_robin"
):

    N = len(texts)
    G = len(encode_fns)
    if N == 0:
        # infer dim from first replica by encoding an empty list safely
        # or just return empty; most callers won't hit this path
        return torch.empty(0, 0)

    # Build shard assignments as (indices, sublist) pairs
    assignments = []
    if shard == "contiguous":
        per = ceil(N / G)
        for g in range(G):
            start = g * per
            end = min(N, (g + 1) * per)
            if start >= end:
                assignments.append(([], []))
            else:
                idxs = list(range(start, end))
                assignments.append((idxs, [texts[i] for i in idxs]))
    elif shard == "round_robin":
        buckets = [[] for _ in range(G)]
        bucket_idxs = [[] for _ in range(G)]
        for i, t in enumerate(texts):
            g = i % G
            bucket_idxs[g].append(i)
            buckets[g].append(t)
        assignments = list(zip(bucket_idxs, buckets))
    else:
        raise ValueError("shard must be 'contiguous' or 'round_robin'")

    # Worker: run encode on its shard
    def _worker(gpu_i: int, idxs: list[int], shard_texts: list[str]):
        if not shard_texts:
            return gpu_i, idxs, None
        embs = encode_fns[gpu_i](shard_texts, batch_size=batch_size)  # CPU tensor [len(idxs), D]
        return gpu_i, idxs, embs

    # Launch all GPUs in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=G) as ex:
        futs = [
            ex.submit(_worker, gpu_i, idxs, shard_texts)
            for gpu_i, (idxs, shard_texts) in enumerate(assignments)
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
        # All shards empty (N==0 handled earlier, but be safe)
        return torch.empty(0, 0)

    # Stitch back into original order
    out = torch.empty(N, emb_dim, dtype=results[0][2].dtype if results[0][2] is not None else torch.float32)
    for _, idxs, embs in results:
        if not idxs:
            continue
        out[idxs] = embs
    return out
