#!/usr/bin/env python3
"""
Ingest pipeline v3 — Step 1: Shard-level embedding with offline vLLM.

Each SLURM array task processes a subset of input parquet files on a single GPU:
  1. Scan assigned parquets, aggregate rows by document, collect unique queries.
  2. Embed all document texts using vLLM offline (LLM.encode) on one GPU.
  3. Write intermediate parquet with canonical per-document schema.
  4. Write a manifest JSON for verification by step 2.

Outputs (in --output-dir/embeddings/):
  shard_{rank:05d}.parquet  — Intermediate embeddings
  shard_{rank:05d}.manifest.json — Verification manifest

Usage:
    python src/ingest_parquet_embed_v3_shard.py \
        --glob "/data/shard-*.parquet" \
        --fields text title \
        --query-field query \
        --embed-model intfloat/multilingual-e5-large-instruct \
        --output-dir /output/embed_index_v3
"""

import argparse
import glob as globlib
import json
import logging
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="v3 shard embedding: offline vLLM on a single GPU"
    )

    g = p.add_argument_group("Input data")
    g.add_argument("--glob", required=True, help="Glob pattern for input parquet files")
    g.add_argument(
        "--fields",
        nargs="+",
        required=True,
        help="Document content field names to embed (concatenated with space). Should NOT include the query column.",
    )
    g.add_argument(
        "--query-field",
        default="query",
        help="Column containing queries to aggregate per document [default: query]",
    )
    g.add_argument(
        "--id-field", default=None, help="Parquet column containing document IDs"
    )

    g = p.add_argument_group("Embedding model (offline vLLM)")
    g.add_argument("--embed-model", required=True, help="HF model path for embeddings")
    g.add_argument(
        "--embed-max-model-len",
        type=int,
        default=512,
        help="Max sequence length for embedding model [default: 512]",
    )
    g.add_argument(
        "--embed-max-num-batched-tokens",
        type=int,
        default=512 * 64,
        help="Max total tokens per batch for vLLM encoding [default: 512*64]",
    )
    g.add_argument(
        "--embed-max-num-seqs",
        type=int,
        default=8192,
        help="Max number of sequences per batch for vLLM encoding [default: 8192]",
    )
    g.add_argument(
        "--embed-gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory fraction [default: 0.90]",
    )
    g.add_argument(
        "--embed-prefix",
        default="",
        help="Prefix to prepend to each text before embedding",
    )
    g.add_argument(
        "--embed-max-tokens",
        type=int,
        default=512,
        help="Whitespace-word limit before the 4096-character cap [default: 512]",
    )
    g.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs")
    g.add_argument("--dtype", default="auto", help="Model dtype [default: auto]")

    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", required=True, help="Base output directory")

    g = p.add_argument_group("Pipeline control")
    g.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Row read batch size from parquet [default: 8192]",
    )
    g.add_argument(
        "--encode-batch-size",
        type=int,
        default=8192,
        help="Texts per vLLM encode call [default: 8192]",
    )

    g = p.add_argument_group("Distributed (SLURM array)")
    g.add_argument(
        "--rank",
        type=int,
        default=None,
        help="This worker's rank (0-based). Defaults to SLURM_ARRAY_TASK_ID or 0.",
    )
    g.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Total number of workers. Defaults to SLURM_ARRAY_TASK_COUNT or 1.",
    )

    g.add_argument("--log-level", default="INFO")

    args = p.parse_args()

    # Validate: query_field must not be in --fields
    if args.query_field in args.fields:
        p.error(
            f"--query-field '{args.query_field}' must not also appear in --fields. "
            f"Use --fields for document content columns only."
        )

    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_base_uuid(raw_id: str) -> str:
    return str(raw_id).rsplit("__s1d_", 1)[0]


def _truncate_text(text, max_words=512):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + " ...(truncated)"
    return text


def aggregate_docs(pq_files, fields, query_field, batch_size, id_field=None):
    """Scan parquet files, group rows by document ID, aggregate unique queries.

    Returns list of dicts (insertion-ordered by first occurrence):
      doc_id, embed_text, pos_text, pos_title, all_query_texts, repr_query_text
    """
    text_field_name = "text" if "text" in fields else fields[0]
    title_field_name = "title" if "title" in fields else None

    # Build deduplicated column list to read
    read_cols_set = set()
    read_cols = []
    for c in list(fields) + [query_field] + ([id_field] if id_field else []):
        if c not in read_cols_set:
            read_cols.append(c)
            read_cols_set.add(c)

    # Ordered storage: doc_id -> index in result list
    doc_order = {}  # doc_id_str -> index
    results = []  # list of dicts

    # Check if query_field exists (set after first file is read)
    query_field_available = None
    running_counter = 0

    for fpath in tqdm(pq_files, desc="Reading parquets", unit="file"):
        pf = pq.ParquetFile(fpath)
        available_columns = set(pf.schema_arrow.names)
        required_columns = {text_field_name} | ({id_field} if id_field else set())
        missing_columns = required_columns - available_columns
        if missing_columns:
            pf.close()
            raise ValueError("Input Parquet {} is missing required document columns {}. "
                             "Check that the input glob does not include embedding outputs.".format(
                                 fpath, sorted(missing_columns)))

        # On first file, check query column availability
        if query_field_available is None:
            if query_field not in available_columns:
                log.warning(
                    "Query field '%s' not found in parquet columns %s. "
                    "All documents will have no queries.",
                    query_field,
                    sorted(available_columns),
                )
                query_field_available = False
            else:
                query_field_available = True

        # Determine actual columns to read for this file
        actual_read_cols = [c for c in read_cols if c in available_columns]

        for rb in pf.iter_batches(batch_size=batch_size, columns=actual_read_cols):
            cols = {f: rb.column(f).to_pylist() for f in actual_read_cols}
            for i in range(rb.num_rows):
                # Determine doc_id
                if id_field and id_field in cols:
                    doc_id = _extract_base_uuid(cols[id_field][i])
                else:
                    doc_id = str(running_counter)
                    running_counter += 1

                if doc_id not in doc_order:
                    # First occurrence: store field values
                    field_values = {}
                    for f in fields:
                        if f in cols:
                            val = cols[f][i]
                            field_values[f] = str(val) if val is not None else ""
                        else:
                            field_values[f] = ""

                    idx = len(results)
                    doc_order[doc_id] = idx
                    results.append({
                        "doc_id": doc_id,
                        "field_values": field_values,
                        "query_list": [],
                        "query_set": set(),
                    })

                # Aggregate query
                if query_field_available and query_field in cols:
                    value = cols[query_field][i]
                    # Compacted generation stores aligned lists of queries.
                    # Preserve each query and its order instead of stringifying a list.
                    for qval in value if isinstance(value, list) else [value]:
                        if qval is None:
                            continue
                        qstr = str(qval).strip()
                        if qstr:
                            entry = results[doc_order[doc_id]]
                            if qstr not in entry["query_set"]:
                                entry["query_list"].append(qstr)
                                entry["query_set"].add(qstr)

    # Build final result list in insertion order
    output = []
    for entry in results:
        fv = entry["field_values"]
        pos_text = fv.get(text_field_name, "")
        pos_title = fv.get(title_field_name, "") if title_field_name else ""
        embed_text = " ".join(fv[f] for f in fields).strip()
        all_query_texts = entry["query_list"]
        repr_query_text = all_query_texts[0] if all_query_texts else ""

        output.append({
            "doc_id": entry["doc_id"],
            "embed_text": embed_text,
            "pos_text": pos_text,
            "pos_title": pos_title,
            "all_query_texts": all_query_texts,
            "repr_query_text": repr_query_text,
        })

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    from vllm import LLM
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Resolve rank / world_size from SLURM or CLI
    rank = args.rank
    if rank is None:
        rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    world_size = args.world_size
    if world_size is None:
        world_size = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    log.info("Rank %d / %d", rank, world_size)

    # Discover and assign input files to this rank
    all_files = sorted(globlib.glob(args.glob, recursive=True))
    if not all_files:
        raise FileNotFoundError(f"No files matched: {args.glob}")
    log.info("Total input files: %d", len(all_files))

    my_files = all_files[rank::world_size]
    if not my_files:
        log.warning("Rank %d has no files to process. Exiting.", rank)
        return
    log.info(
        "Rank %d processing %d file(s): %s … %s",
        rank,
        len(my_files),
        my_files[0],
        my_files[-1],
    )

    # Output paths
    embed_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(embed_dir, exist_ok=True)
    shard_parquet_path = os.path.join(embed_dir, f"shard_{rank:05d}.parquet")
    manifest_path = os.path.join(embed_dir, f"shard_{rank:05d}.manifest.json")

    # Check if already done
    if os.path.exists(manifest_path) and os.path.exists(shard_parquet_path):
        log.info("Shard %d already completed (manifest exists). Skipping.", rank)
        return

    # -- Phase 1: Collect and aggregate documents ----------------------------
    log.info("Phase 1: Reading and aggregating documents …")
    t0 = time.time()

    docs = aggregate_docs(
        my_files, args.fields, args.query_field, args.batch_size, args.id_field
    )

    num_docs = len(docs)
    total_queries = sum(len(d["all_query_texts"]) for d in docs)
    log.info(
        "Phase 1 done: %d docs, %d total queries in %.1fs",
        num_docs,
        total_queries,
        time.time() - t0,
    )

    if num_docs == 0:
        log.warning("Rank %d: no documents after aggregation. Writing empty manifest.", rank)
        manifest = {
            "rank": rank,
            "world_size": world_size,
            "input_files": [os.path.abspath(f) for f in my_files],
            "num_docs": 0,
            "num_queries": 0,
            "embedding_dim": 0,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return

    # Prepare texts for embedding
    all_texts = [
        args.embed_prefix
        + _truncate_text(d["embed_text"], args.embed_max_tokens)[:4096]
        for d in docs
    ]

    # -- Phase 2: Embed with offline vLLM ------------------------------------
    log.info("Phase 2: Loading embedding model …")
    t0 = time.time()

    engine_args = {}
    if "gemma" in args.embed_model.lower():
        engine_args["block_size"] = 64

    llm = LLM(
        model=args.embed_model,
        runner="pooling",
        max_model_len=args.embed_max_model_len,
        max_num_batched_tokens=args.embed_max_num_batched_tokens,
        max_num_seqs=args.embed_max_num_seqs,
        gpu_memory_utilization=args.embed_gpu_memory_utilization,
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_chunked_prefill=True,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        **engine_args,
    )
    log.info("Model loaded in %.1fs", time.time() - t0)

    # Encode in batches to avoid OOM on very large shards
    log.info("Phase 2: Encoding %d texts …", num_docs)
    t_encode = time.time()

    all_embeddings = []
    for start in range(0, num_docs, args.encode_batch_size):
        end = min(start + args.encode_batch_size, num_docs)
        batch_texts = all_texts[start:end]
        log.info("  Encoding batch %d–%d / %d …", start, end, num_docs)
        outputs = llm.embed(batch_texts)
        batch_vecs = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
        all_embeddings.append(batch_vecs)

    embeddings = np.concatenate(all_embeddings, axis=0)
    ndim = embeddings.shape[1]
    encode_time = time.time() - t_encode
    log.info(
        "Encoding done: %d vectors, dim=%d, %.1fs (%.0f docs/s)",
        num_docs,
        ndim,
        encode_time,
        num_docs / max(encode_time, 1e-9),
    )

    # Free GPU memory
    del llm
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    # -- Phase 3: Write intermediate parquet ---------------------------------
    log.info("Phase 3: Writing intermediate parquet …")
    t0 = time.time()

    # Build schema
    schema = pa.schema([
        pa.field("doc_id_local", pa.int64()),
        pa.field("doc_id_str", pa.string()),
        pa.field("pos_text", pa.string()),
        pa.field("pos_title", pa.string()),
        pa.field("all_query_texts", pa.list_(pa.string())),
        pa.field("repr_query_text", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), ndim)),
    ])

    # Build columns
    columns = {
        "doc_id_local": pa.array(list(range(num_docs)), type=pa.int64()),
        "doc_id_str": pa.array([d["doc_id"] for d in docs], type=pa.string()),
        "pos_text": pa.array([d["pos_text"] for d in docs], type=pa.string()),
        "pos_title": pa.array([d["pos_title"] for d in docs], type=pa.string()),
        "all_query_texts": pa.array(
            [d["all_query_texts"] for d in docs], type=pa.list_(pa.string())
        ),
        "repr_query_text": pa.array(
            [d["repr_query_text"] for d in docs], type=pa.string()
        ),
    }
    # Embedding as fixed-size list
    flat = embeddings.ravel().tolist()
    columns["embedding"] = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.float32()), ndim
    )

    table = pa.table(columns, schema=schema)

    # Write with row groups for efficient reading
    tmp_path = shard_parquet_path + ".tmp"
    pq.write_table(table, tmp_path, row_group_size=100_000)
    os.replace(tmp_path, shard_parquet_path)

    log.info(
        "Parquet written: %s (%d rows, %.1fs)",
        shard_parquet_path,
        num_docs,
        time.time() - t0,
    )

    # -- Phase 4: Write manifest ---------------------------------------------
    manifest = {
        "rank": rank,
        "world_size": world_size,
        "input_files": [os.path.abspath(f) for f in my_files],
        "num_docs": num_docs,
        "num_queries": total_queries,
        "embedding_dim": ndim,
        "encode_time_s": round(encode_time, 2),
        "docs_per_sec": round(num_docs / max(encode_time, 1e-9), 1),
        "shard_parquet": os.path.abspath(shard_parquet_path),
    }
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_manifest, manifest_path)

    log.info("Manifest written: %s", manifest_path)
    log.info("Rank %d done. %d docs embedded, %d queries aggregated.", rank, num_docs, total_queries)


if __name__ == "__main__":
    main()
