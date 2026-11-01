#!/usr/bin/env python3
"""
Generate hard negatives v4 — Ray Data + Ray Serve LLM pipeline.

Two-pass approach:
  Pass 1 — Iterate canonical shards sequentially → batch USearch retrieval
           → save top-k neighbors + query payload per document to parquet.
  Pass 2 — Ray Data pipeline: read intermediates → DataJoiner (join
           neighbor texts, expand to per-document pairs) → Ray Serve LLM
           reranker (ScoreRequest) → Aggregator → write parquet.

Key difference from v3: scores negatives once per document (using a
representative query), then shares those scores across all queries for
that document.  ~4x fewer reranker calls.

Usage:
    python src/generate_hard_negatives_v4.py \\
        --embeddings-dir /data/embed_index_v3/embeddings \\
        --index-dir /data/embed_index_v3 \\
        --reranker-model BAAI/bge-reranker-v2-m3 \\
        --output-dir /output/hard_negatives
"""

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm
from usearch.index import Index

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


def _format_bytes(num_bytes):
    """Format byte counts for telemetry logs."""
    if num_bytes is None:
        return "?"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0


def _print_telemetry(message, *args):
    """Emit Ray actor telemetry reliably to worker logs."""
    if args:
        message = message % args
    print(message, flush=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list_arg(value):
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    items = [item for item in items if item]
    if not items:
        return None
    return [int(item) for item in items]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate hard negatives v4: Ray Data + Ray Serve LLM reranking"
    )

    g = p.add_argument_group("Data sources")
    g.add_argument(
        "--embeddings-dir",
        required=True,
        help="Directory containing shard_*.parquet with doc_id_str + embedding columns",
    )
    g.add_argument(
        "--index-dir",
        required=True,
        help="Directory containing index.usearch and doc_texts/",
    )

    g = p.add_argument_group("Embedding retrieval")
    g.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Nearest neighbors to retrieve per document [default: 100]",
    )
    g.add_argument(
        "--skip-top-n",
        type=int,
        default=0,
        help="Skip the top-N nearest neighbors (likely false negatives). "
        "Their doc IDs are preserved in the output. Retrieval fetches topk + skip_top_n. [default: 0]",
    )
    g.add_argument(
        "--search-threads",
        type=int,
        default=16,
        help="USearch search threads [default: 16]",
    )

    g = p.add_argument_group("Cross-encoder model (Ray Serve LLM)")
    g.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
        help="HF model path for reranker [default: BAAI/bge-reranker-v2-m3]",
    )
    g.add_argument(
        "--reranker-max-model-len",
        type=int,
        default=8192,
        help="Max sequence length for reranker model [default: 8192]",
    )
    g.add_argument(
        "--reranker-gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory fraction for reranker model [default: 0.90]",
    )
    g.add_argument(
        "--reranker-tp-size",
        type=int,
        default=1,
        help="Tensor-parallel size for reranker model [default: 1]",
    )
    g.add_argument(
        "--reranker-max-num-batched-tokens",
        type=int,
        default=16384,
        help="Max batched tokens for reranker [default: 16384]",
    )
    g.add_argument(
        "--reranker-max-num-seqs",
        type=int,
        default=256,
        help="Max concurrent sequences for reranker [default: 256]",
    )
    g.add_argument(
        "--reranker-chat-template",
        default=None,
        help="Path to a Jinja2 score/chat template for the reranker (passed as vLLM --chat-template)",
    )
    g.add_argument(
        "--reranker-concurrency",
        type=int,
        default=4,
        help="Number of Ray Serve reranker replicas [default: 4]",
    )
    g.add_argument(
        "--reranker-batch-size",
        type=int,
        default=64,
        help="Batch size for score processor [default: 64]",
    )
    g.add_argument(
        "--store-raw-reranker-scores",
        action="store_true",
        help=(
            "Also store raw reranker logits by requesting unactivated vLLM scores. "
            "Use for one-label sigmoid rerankers such as bge-reranker-v2-gemma-seq-cls."
        ),
    )
    g.add_argument(
        "--reranker-cudagraph-capture-sizes",
        default=None,
        help="Comma-separated cudagraph capture sizes to pass to vLLM. "
        "If unset, vLLM chooses sizes automatically.",
    )
    g.add_argument(
        "--reranker-max-cudagraph-capture-size",
        type=int,
        default=None,
        help="Upper bound for vLLM cudagraph capture sizes. "
        "Useful to cap warmup memory near the real reranker batch size.",
    )

    g = p.add_argument_group("vLLM general")
    g.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Total GPUs available (auto-detected if not set)",
    )
    g.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (useful for debugging)",
    )
    g.add_argument("--dtype", default="auto", help="Model dtype [default: auto]")
    g.add_argument(
        "--ray-object-store-memory-gb",
        type=float,
        default=None,
        help="Explicit Ray object store size in GiB. "
        "Use this to avoid over-reserving host RAM for Plasma on large-memory nodes.",
    )

    g = p.add_argument_group("Pipeline control")
    g.add_argument("--output-dir", required=True, help="Directory for output parquets")
    g.add_argument(
        "--checkpoint-file",
        default=None,
        help="Override checkpoint path (default: <output-dir>/checkpoint_rank{rank:04d}.json)",
    )
    g.add_argument(
        "--pass2-chunk-size",
        type=int,
        default=0,
        help=(
            "If > 0, process Pass 2 in chunks of this many intermediate "
            "document rows and checkpoint each chunk. The default 0 keeps "
            "the original whole-shard Pass 2 behavior."
        ),
    )
    g.add_argument(
        "--pass2-chunk-block-size",
        type=int,
        default=512,
        help=(
            "When --pass2-chunk-size is enabled, split each checkpoint chunk "
            "into Ray input blocks of this many document rows before handing "
            "them to Ray. This avoids serializing one very large Arrow object "
            "per chunk; DataJoiner can still use --joiner-batch-size to "
            "coalesce those blocks into larger batches. [default: 512]"
        ),
    )
    g.add_argument(
        "--materialized-output-basename",
        default="hard_negatives.parquet",
        help=(
            "Final per-shard parquet filename written by "
            "materialize_chunked_hard_negatives.py. If this file and "
            "_MATERIALIZED_SUCCESS are present in a shard output directory, "
            "the shard is treated as Pass 2 complete on relaunch. "
            "[default: hard_negatives.parquet]"
        ),
    )
    g.add_argument("--log-level", default="INFO")

    g = p.add_argument_group("DataJoiner")
    g.add_argument(
        "--joiner-batch-size",
        type=int,
        default=512,
        help="Batch size for DataJoiner map_batches [default: 512]",
    )
    g.add_argument(
        "--joiner-concurrency",
        type=int,
        default=2,
        help="Number of DataJoiner actor replicas [default: 2]",
    )
    g.add_argument(
        "--joiner-log-every-batches",
        type=int,
        default=1,
        help=(
            "Log DataJoiner per-actor progress every N input batches. "
            "Set to 0 to disable periodic Joiner progress logs. [default: 1]"
        ),
    )
    g.add_argument(
        "--doc-text-fetch-concurrency",
        type=int,
        default=1,
        help=(
            "Number of doc_texts shards to fetch in parallel inside each DataJoiner. "
            "Keep at 1 when using multiple DataJoiner actors; try 2-8 when "
            "JOINER_CONCURRENCY=1 and text lookup is I/O-bound. [default: 1]"
        ),
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
    g.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging, limit to first 2 shards and 1000 docs per shard",
    )

    args = p.parse_args()
    args.reranker_cudagraph_capture_sizes = _parse_int_list_arg(
        args.reranker_cudagraph_capture_sizes
    )
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_text(text, max_words=512):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + " ...(truncated)"
    return text


def _detect_num_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


# ---------------------------------------------------------------------------
# Embedding shard utilities
# ---------------------------------------------------------------------------


def list_embedding_shards(embeddings_dir):
    """Return sorted list of shard parquet paths."""
    files = sorted(
        f
        for f in os.listdir(embeddings_dir)
        if f.endswith(".parquet") and f.startswith("shard_")
    )
    return [os.path.join(embeddings_dir, f) for f in files]


def compute_shard_offsets(embeddings_dir, shard_paths):
    """Compute cumulative doc_id_int offsets from manifest files or parquet metadata."""
    offsets = [0]
    for path in shard_paths:
        manifest_path = path.replace(".parquet", ".manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                n = json.load(f)["num_docs"]
        else:
            n = pq.read_metadata(path).num_rows
        offsets.append(offsets[-1] + n)
    return offsets


def compute_query_offsets(embeddings_dir, shard_paths):
    """Compute cumulative query ID offsets from manifest num_queries."""
    offsets = [0]
    for path in shard_paths:
        manifest_path = path.replace(".parquet", ".manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                n = json.load(f).get("num_queries", 0)
        else:
            table = pq.read_table(path, columns=["all_query_texts"])
            n = sum(len(q) for q in table.column("all_query_texts").to_pylist())
        offsets.append(offsets[-1] + n)
    return offsets


def read_canonical_shard(shard_path):
    """Read canonical shard: embeddings + query payload."""
    table = pq.read_table(shard_path)
    doc_id_strs = table.column("doc_id_str").to_pylist()

    emb_col = table.column("embedding")
    embeddings = (
        emb_col.combine_chunks()
        .values.to_numpy(zero_copy_only=False)
        .reshape(len(emb_col), -1)
        .astype(np.float32)
    )

    query_payload = {
        "all_query_texts": table.column("all_query_texts").to_pylist(),
        "repr_query_text": table.column("repr_query_text").to_pylist(),
        "pos_text": table.column("pos_text").to_pylist(),
        "pos_title": table.column("pos_title").to_pylist(),
    }

    return doc_id_strs, embeddings, query_payload


# ---------------------------------------------------------------------------
# Memory-mapped doc texts reader (Arrow IPC)
# ---------------------------------------------------------------------------


class MmapDocTextsReader:
    """Memory-mapped doc texts reader using Arrow IPC files.

    Converts parquet shards to Arrow IPC format on first use (one-time cost),
    then memory-maps them.
    """

    def __init__(self, texts_dir_or_file, should_log=True, text_fetch_concurrency=1):
        self._tables = {}
        self._fields = None
        self._should_log = should_log
        self._text_fetch_concurrency = max(1, int(text_fetch_concurrency))
        self._pid = os.getpid()
        self._ensure_table_calls = 0
        self._table_cache_hits = 0
        self._table_cache_misses = 0
        self._conversion_count = 0
        self._arrow_load_count = 0
        self._arrow_load_rows = 0
        self._arrow_load_bytes = 0
        self._bulk_calls = 0
        self._bulk_docs_requested = 0
        self._bulk_docs_returned = 0
        self._bulk_shards_touched = 0
        self._field_take_calls = 0
        self._field_take_rows = 0
        self._fields_missing = 0

        if os.path.isdir(texts_dir_or_file):
            meta_path = os.path.join(texts_dir_or_file, "meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            self._dir = texts_dir_or_file
            self._shard_size = meta["shard_size"]
            self._num_shards = meta["num_shards"]
            self._mode = "sharded"
        else:
            self._dir = os.path.dirname(texts_dir_or_file)
            self._single_file = texts_dir_or_file
            self._mode = "single"
            self._shard_size = pq.read_metadata(texts_dir_or_file).num_rows
            self._num_shards = 1

        _print_telemetry(
            "MmapDocTextsReader: mode=%s, %d shards, shard_size=%d, pid=%d, "
            "text_fetch_concurrency=%d",
            self._mode,
            self._num_shards,
            self._shard_size,
            self._pid,
            self._text_fetch_concurrency,
        )

    def _parquet_path(self, shard_idx):
        if self._mode == "sharded":
            return os.path.join(self._dir, f"shard_{shard_idx:05d}.parquet")
        return self._single_file

    def _arrow_path(self, shard_idx):
        if self._mode == "sharded":
            return os.path.join(self._dir, f"shard_{shard_idx:05d}.arrow")
        return self._single_file.replace(".parquet", ".arrow")

    def _ensure_table(self, shard_idx):
        self._ensure_table_calls += 1
        if shard_idx in self._tables:
            self._table_cache_hits += 1
            table = self._tables[shard_idx]
            if self._should_log:
                _print_telemetry(
                    "MmapDocTextsReader cache hit: pid=%d shard=%d rows=%d cols=%d "
                    "table_bytes=%s cached_shards=%d hits=%d misses=%d",
                    self._pid,
                    shard_idx,
                    table.num_rows,
                    table.num_columns,
                    _format_bytes(table.nbytes),
                    len(self._tables),
                    self._table_cache_hits,
                    self._table_cache_misses,
                )
            return table

        self._table_cache_misses += 1
        arrow_path = self._arrow_path(shard_idx)
        ensure_t0 = time.time()

        if not os.path.exists(arrow_path):
            parquet_path = self._parquet_path(shard_idx)
            parquet_bytes = os.path.getsize(parquet_path)
            convert_t0 = time.time()
            _print_telemetry(
                "MmapDocTextsReader convert start: pid=%d shard=%d parquet=%s "
                "parquet_bytes=%s arrow=%s",
                self._pid,
                shard_idx,
                parquet_path,
                _format_bytes(parquet_bytes),
                arrow_path,
            )
            read_t0 = time.time()
            table = pq.read_table(parquet_path)
            read_s = time.time() - read_t0
            if "doc_id_int" in table.column_names:
                table = table.drop(["doc_id_int"])
            if self._fields is None:
                self._fields = table.column_names
            tmp_path = arrow_path + ".tmp"
            write_t0 = time.time()
            with pa.OSFile(tmp_path, "wb") as f:
                writer = pa.ipc.new_file(f, table.schema)
                writer.write_table(table)
                writer.close()
            os.replace(tmp_path, arrow_path)
            write_s = time.time() - write_t0
            arrow_bytes = os.path.getsize(arrow_path)
            self._conversion_count += 1
            _print_telemetry(
                "MmapDocTextsReader convert done: pid=%d shard=%d rows=%d cols=%d "
                "table_bytes=%s parquet_bytes=%s arrow_bytes=%s read_s=%.2f "
                "write_s=%.2f total_s=%.2f conversions=%d",
                self._pid,
                shard_idx,
                table.num_rows,
                table.num_columns,
                _format_bytes(table.nbytes),
                _format_bytes(parquet_bytes),
                _format_bytes(arrow_bytes),
                read_s,
                write_s,
                time.time() - convert_t0,
                self._conversion_count,
            )
            del table

        arrow_bytes = os.path.getsize(arrow_path)
        load_t0 = time.time()
        _print_telemetry(
            "MmapDocTextsReader mmap load start: pid=%d shard=%d arrow=%s "
            "arrow_bytes=%s cached_shards=%d ensure_calls=%d hits=%d misses=%d",
            self._pid,
            shard_idx,
            arrow_path,
            _format_bytes(arrow_bytes),
            len(self._tables),
            self._ensure_table_calls,
            self._table_cache_hits,
            self._table_cache_misses,
        )
        mmap = pa.memory_map(arrow_path, "r")
        reader = pa.ipc.open_file(mmap)
        table = reader.read_all()
        self._tables[shard_idx] = table
        self._arrow_load_count += 1
        self._arrow_load_rows += table.num_rows
        self._arrow_load_bytes += table.nbytes

        if self._fields is None:
            self._fields = table.column_names

        _print_telemetry(
            "MmapDocTextsReader mmap load done: pid=%d shard=%d rows=%d cols=%d "
            "fields=%s table_bytes=%s arrow_bytes=%s load_s=%.2f ensure_s=%.2f "
            "cached_shards=%d loads=%d loaded_rows=%d loaded_table_bytes=%s",
            self._pid,
            shard_idx,
            table.num_rows,
            table.num_columns,
            ",".join(table.column_names),
            _format_bytes(table.nbytes),
            _format_bytes(arrow_bytes),
            time.time() - load_t0,
            time.time() - ensure_t0,
            len(self._tables),
            self._arrow_load_count,
            self._arrow_load_rows,
            _format_bytes(self._arrow_load_bytes),
        )
        return table

    def _fetch_texts_for_shard(self, bulk_call_id, si, shard_pos, total_shards, items, fields):
        """Fetch all requested rows for one doc_texts shard."""
        items = sorted(items, key=lambda item: item[1])
        shard_t0 = time.time()
        ensure_t0 = time.time()
        table = self._ensure_table(si)
        ensure_s = time.time() - ensure_t0
        doc_int_list = [x[0] for x in items]
        row_values = [x[1] for x in items]
        row_indices = pa.array(row_values, type=pa.int32())
        row_span = f"{min(row_values)}..{max(row_values)}" if row_values else "empty"

        _print_telemetry(
            "MmapDocTextsReader bulk shard start: pid=%d call=%d shard=%d "
            "position=%d/%d requested_rows=%d row_span=%s table_rows=%d "
            "table_cols=%d ensure_s=%.2f cached_shards=%d",
            self._pid,
            bulk_call_id,
            si,
            shard_pos,
            total_shards,
            len(items),
            row_span,
            table.num_rows,
            table.num_columns,
            ensure_s,
            len(self._tables),
        )

        field_values = {}
        shard_take_s = 0.0
        shard_to_py_s = 0.0
        shard_cast_s = 0.0
        missing_fields = []
        for field in fields:
            field_t0 = time.time()
            if field in table.column_names:
                col = table.column(field)
                original_type = str(col.type)
                cast_s = 0.0
                casted = False
                if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
                    cast_t0 = time.time()
                    col = col.cast(pa.large_string())
                    cast_s = time.time() - cast_t0
                    casted = True
                take_t0 = time.time()
                taken_arr = col.take(row_indices)
                take_s = time.time() - take_t0
                to_py_t0 = time.time()
                taken = taken_arr.to_pylist()
                to_py_s = time.time() - to_py_t0
                null_count = sum(1 for value in taken if value is None)
                empty_count = sum(1 for value in taken if value == "")
                shard_cast_s += cast_s
                shard_take_s += take_s
                shard_to_py_s += to_py_s
                self._field_take_calls += 1
                self._field_take_rows += len(items)
                _print_telemetry(
                    "MmapDocTextsReader field take done: pid=%d call=%d "
                    "shard=%d field=%s rows=%d nulls=%d empty=%d "
                    "type=%s casted=%s cast_s=%.3f take_s=%.3f "
                    "to_py_s=%.3f total_s=%.3f",
                    self._pid,
                    bulk_call_id,
                    si,
                    field,
                    len(items),
                    null_count,
                    empty_count,
                    original_type,
                    casted,
                    cast_s,
                    take_s,
                    to_py_s,
                    time.time() - field_t0,
                )
                field_values[field] = taken
            else:
                missing_fields.append(field)
                self._fields_missing += 1
                _print_telemetry(
                    "WARNING MmapDocTextsReader missing field: pid=%d call=%d "
                    "shard=%d field=%s available_fields=%s rows=%d",
                    self._pid,
                    bulk_call_id,
                    si,
                    field,
                    ",".join(table.column_names),
                    len(items),
                )
                field_values[field] = [""] * len(items)

        empty_records = 0
        shard_result = {}
        for idx, di in enumerate(doc_int_list):
            values = {f: (field_values[f][idx] or "") for f in fields}
            if not any(values.values()):
                empty_records += 1
            shard_result[di] = values

        _print_telemetry(
            "MmapDocTextsReader bulk shard done: pid=%d call=%d shard=%d "
            "rows=%d result_docs=%d empty_records=%d missing_fields=%s "
            "cast_s=%.3f take_s=%.3f to_py_s=%.3f shard_s=%.2f",
            self._pid,
            bulk_call_id,
            si,
            len(items),
            len(doc_int_list),
            empty_records,
            ",".join(missing_fields) if missing_fields else "-",
            shard_cast_s,
            shard_take_s,
            shard_to_py_s,
            time.time() - shard_t0,
        )
        return shard_result

    def get_texts_bulk(self, doc_ints, fields):
        """Batch-fetch texts for many doc_ints at once using Arrow take()."""
        call_t0 = time.time()
        self._bulk_calls += 1
        fields = list(fields)
        doc_ints = [int(di) for di in doc_ints]
        shard_groups = {}
        for di in doc_ints:
            si = di // self._shard_size
            row = di % self._shard_size
            shard_groups.setdefault(si, []).append((di, row))

        requested_docs = len(doc_ints)
        unique_docs = len(set(doc_ints))
        duplicate_docs = requested_docs - unique_docs
        shard_ids = sorted(shard_groups)
        doc_span = (
            f"{min(doc_ints)}..{max(doc_ints)}" if doc_ints else "empty"
        )
        shard_span = (
            f"{shard_ids[0]}..{shard_ids[-1]}" if shard_ids else "empty"
        )
        start_msg = (
            "MmapDocTextsReader bulk fetch start: pid=%d call=%d docs=%d "
            "unique_docs=%d duplicate_entries=%d fields=%s shards=%d "
            "shard_span=%s doc_span=%s cached_shards=%d text_fetch_concurrency=%d"
        ) % (
            self._pid,
            self._bulk_calls,
            requested_docs,
            unique_docs,
            duplicate_docs,
            ",".join(fields),
            len(shard_groups),
            shard_span,
            doc_span,
            len(self._tables),
            self._text_fetch_concurrency,
        )
        _print_telemetry(start_msg)

        result = {}
        shard_tasks = [
            (shard_pos, si, shard_groups[si])
            for shard_pos, si in enumerate(shard_ids, start=1)
        ]
        fetch_concurrency = min(self._text_fetch_concurrency, len(shard_tasks))
        if fetch_concurrency > 1:
            _print_telemetry(
                "MmapDocTextsReader parallel shard fetch: pid=%d call=%d "
                "workers=%d shards=%d",
                self._pid,
                self._bulk_calls,
                fetch_concurrency,
                len(shard_tasks),
            )
            with ThreadPoolExecutor(max_workers=fetch_concurrency) as executor:
                futures = [
                    executor.submit(
                        self._fetch_texts_for_shard,
                        self._bulk_calls,
                        si,
                        shard_pos,
                        len(shard_ids),
                        items,
                        fields,
                    )
                    for shard_pos, si, items in shard_tasks
                ]
                for future in as_completed(futures):
                    result.update(future.result())
        else:
            for shard_pos, si, items in shard_tasks:
                result.update(
                    self._fetch_texts_for_shard(
                        self._bulk_calls,
                        si,
                        shard_pos,
                        len(shard_ids),
                        items,
                        fields,
                    )
                )

        elapsed = time.time() - call_t0
        self._bulk_docs_requested += requested_docs
        self._bulk_docs_returned += len(result)
        self._bulk_shards_touched += len(shard_groups)
        done_msg = (
            "MmapDocTextsReader bulk fetch done: pid=%d call=%d docs=%d "
            "unique_docs=%d returned=%d shards=%d elapsed_s=%.2f docs_per_s=%.1f "
            "totals: calls=%d docs_requested=%d docs_returned=%d "
            "shards_touched=%d ensure_calls=%d cache_hits=%d cache_misses=%d "
            "field_take_calls=%d field_take_rows=%d missing_fields=%d "
            "loaded_shards=%d loaded_rows=%d loaded_table_bytes=%s"
        ) % (
            self._pid,
            self._bulk_calls,
            requested_docs,
            unique_docs,
            len(result),
            len(shard_groups),
            elapsed,
            requested_docs / max(elapsed, 1e-9),
            self._bulk_calls,
            self._bulk_docs_requested,
            self._bulk_docs_returned,
            self._bulk_shards_touched,
            self._ensure_table_calls,
            self._table_cache_hits,
            self._table_cache_misses,
            self._field_take_calls,
            self._field_take_rows,
            self._fields_missing,
            self._arrow_load_count,
            self._arrow_load_rows,
            _format_bytes(self._arrow_load_bytes),
        )
        _print_telemetry(done_msg)

        return result


def preconvert_doc_texts_to_arrow(index_dir):
    """Convert all doc_texts parquet shards to Arrow IPC before Ray processing.

    This avoids redundant conversion inside Ray actor workers.
    """
    reader = open_doc_texts_reader(index_dir)
    log.info(
        "Pre-converting doc_texts to Arrow IPC (%d shards) ...", reader._num_shards
    )
    t0 = time.time()
    converted = 0
    for shard_idx in range(reader._num_shards):
        arrow_path = reader._arrow_path(shard_idx)
        if os.path.exists(arrow_path):
            continue
        parquet_path = reader._parquet_path(shard_idx)
        log.info("  Converting %s -> %s ...", parquet_path, arrow_path)
        table = pq.read_table(parquet_path)
        if "doc_id_int" in table.column_names:
            table = table.drop(["doc_id_int"])
        tmp_path = arrow_path + ".tmp"
        with pa.OSFile(tmp_path, "wb") as f:
            writer = pa.ipc.new_file(f, table.schema)
            writer.write_table(table)
            writer.close()
        os.replace(tmp_path, arrow_path)
        del table
        converted += 1
    log.info(
        "Pre-conversion done: %d/%d shards converted in %.1fs",
        converted,
        reader._num_shards,
        time.time() - t0,
    )


def open_doc_texts_reader(index_dir, text_fetch_concurrency=1):
    texts_dir = os.path.join(index_dir, "doc_texts")
    texts_file = os.path.join(index_dir, "doc_texts.parquet")
    if os.path.isdir(texts_dir) and os.path.exists(
        os.path.join(texts_dir, "meta.json")
    ):
        log.info("Using mmap doc_texts reader from %s", texts_dir)
        return MmapDocTextsReader(
            texts_dir, text_fetch_concurrency=text_fetch_concurrency
        )
    elif os.path.isfile(texts_file):
        log.info("Using mmap doc_texts reader from %s (single file)", texts_file)
        return MmapDocTextsReader(
            texts_file, text_fetch_concurrency=text_fetch_concurrency
        )
    else:
        raise FileNotFoundError(
            f"No doc_texts found in {index_dir}. Expected either "
            f"'{texts_dir}/' directory or '{texts_file}' file."
        )


# ---------------------------------------------------------------------------
# Phase 1 — Document-centric USearch retrieval
# ---------------------------------------------------------------------------


def phase1_retrieve(
    embeddings,
    self_doc_ints,
    usearch_index,
    topk,
    num_docs,
    threads,
    skip_top_n=0,
    batch_size=10_000,
):
    """Batch-search USearch and filter self-matches.

    Returns (all_neighbors, all_skipped) where each is a list of
    [(doc_int, distance), ...] per document. ``all_skipped`` contains
    the first ``skip_top_n`` neighbors (likely false negatives) that
    are excluded from ``all_neighbors``.
    """
    fetch_k = topk + skip_top_n
    n = len(self_doc_ints)
    num_batches = math.ceil(n / batch_size)
    log.info(
        "  Starting USearch search: %d embeddings in %d batches of %d (topk=%d, skip_top_n=%d, fetch_k=%d)",
        n,
        num_batches,
        batch_size,
        topk,
        skip_top_n,
        fetch_k,
    )
    t0 = time.time()

    all_neighbors = []
    all_skipped = []
    for batch_idx in tqdm(range(num_batches), desc="USearch batches", unit="batch"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch_embs = embeddings[start:end]
        batch_ints = self_doc_ints[start:end]

        matches = usearch_index.search(batch_embs, fetch_k + 1, threads=threads)

        for i in range(end - start):
            # USearch returns Matches rather than BatchMatches for a singleton
            # input, even when the query array has shape (1, dimensions).
            query_matches = matches[i] if hasattr(matches, "counts") else matches
            count = (
                int(matches.counts[i])
                if hasattr(matches, "counts")
                else len(query_matches.keys)
            )
            valid = []
            self_int = batch_ints[i]
            for j in range(min(count, fetch_k + 1)):
                doc_int = int(query_matches.keys[j])
                dist = float(query_matches.distances[j])
                if doc_int == self_int:
                    continue
                if np.isnan(dist) or doc_int < 0 or doc_int >= num_docs:
                    continue
                valid.append((doc_int, dist))
                if len(valid) >= fetch_k:
                    break
            all_skipped.append(valid[:skip_top_n])
            all_neighbors.append(valid[skip_top_n:])

    log.info(
        "  USearch search done in %.1fs (%.0f docs/s)",
        time.time() - t0,
        n / max(time.time() - t0, 1e-9),
    )
    return all_neighbors, all_skipped


# ---------------------------------------------------------------------------
# Pass 1 intermediate I/O (parquet)
# ---------------------------------------------------------------------------

INTERMEDIATE_SCHEMA = pa.schema(
    [
        pa.field("doc_id_str", pa.string()),
        pa.field("doc_id_int", pa.int64()),
        pa.field(
            "neighbors",
            pa.list_(
                pa.struct(
                    [
                        pa.field("doc_int", pa.int64()),
                        pa.field("distance", pa.float64()),
                    ]
                )
            ),
        ),
        pa.field(
            "skipped_doc_ids",
            pa.list_(
                pa.struct(
                    [
                        pa.field("doc_id", pa.int64()),
                        pa.field("score", pa.float64()),
                    ]
                )
            ),
        ),
        pa.field("all_query_texts", pa.list_(pa.string())),
        pa.field("all_qids", pa.list_(pa.int64())),
        pa.field("repr_query_text", pa.string()),
        pa.field("pos_text", pa.string()),
        pa.field("pos_title", pa.string()),
    ]
)


def write_intermediate_parquet(intermediate, path):
    """Write Pass 1 intermediate results as parquet."""
    rows = []
    for doc in intermediate:
        rows.append(
            {
                "doc_id_str": doc["doc_id_str"],
                "doc_id_int": doc["doc_id_int"],
                "neighbors": [
                    {"doc_int": n["doc_int"], "distance": n["distance"]}
                    for n in doc["neighbors"]
                ],
                "skipped_doc_ids": [
                    {"doc_id": s["doc_id"], "score": s["score"]}
                    for s in doc.get("skipped_doc_ids", [])
                ],
                "all_query_texts": doc["all_query_texts"],
                "all_qids": doc["all_qids"],
                "repr_query_text": doc["repr_query_text"],
                "pos_text": doc["pos_text"],
                "pos_title": doc["pos_title"],
            }
        )
    table = pa.Table.from_pylist(rows, schema=INTERMEDIATE_SCHEMA)
    tmp = path + ".tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    log.info("  Wrote %d intermediate rows -> %s", len(rows), path)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _checkpoint_path(args):
    if args.checkpoint_file:
        return args.checkpoint_file
    return os.path.join(args.output_dir, f"checkpoint_rank{args.rank:04d}.json")


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"pass1_completed": [], "pass2_completed": []}


def save_checkpoint(ckpt, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, path)


def _append_unique(items, item):
    if item not in items:
        items.append(item)


def _atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# DataJoiner actor — join neighbor texts + expand to per-document pairs
# ---------------------------------------------------------------------------


class DataJoiner:
    """Joins intermediate results with neighbor texts, expands to per-document pairs.
    Reads query payload directly from the batch (no external lookup).
    """

    def __init__(
        self,
        index_dir: str,
        shard_id: int | None = None,
        log_every_batches: int = 1,
        doc_text_fetch_concurrency: int = 1,
    ):
        self._doc_reader = open_doc_texts_reader(
            index_dir, text_fetch_concurrency=doc_text_fetch_concurrency
        )
        self._shard_id = shard_id
        self._log_every_batches = max(0, int(log_every_batches))
        self._doc_text_fetch_concurrency = max(1, int(doc_text_fetch_concurrency))
        self._pid = os.getpid()
        self._t0 = time.time()
        self._batches_seen = 0
        self._docs_seen = 0
        self._pairs_emitted = 0
        self._docs_with_queries = 0
        self._docs_without_queries = 0
        self._neighbor_refs_seen = 0
        self._unique_neighbors_requested = 0
        self._neighbor_texts_returned = 0
        self._missing_neighbor_texts = 0
        self._empty_candidate_texts = 0
        self._positive_pairs = 0
        self._negative_pairs = 0
        self._lookup_s = 0.0
        self._build_s = 0.0
        init_msg = (
            "DataJoiner initialized (index_dir=%s, shard=%s, pid=%d, "
            "log_every_batches=%d, doc_text_fetch_concurrency=%d)",
            index_dir,
            self._shard_id if self._shard_id is not None else "?",
            self._pid,
            self._log_every_batches,
            self._doc_text_fetch_concurrency,
        )
        _print_telemetry(init_msg[0], *init_msg[1:])

    def __call__(self, batch: dict) -> dict:
        batch_t0 = time.time()
        doc_id_strs = batch["doc_id_str"]
        doc_id_ints = batch["doc_id_int"]
        neighbors_list = batch["neighbors"]
        all_query_texts_list = batch["all_query_texts"]
        all_qids_list = batch["all_qids"]
        skipped_doc_ids_list = batch["skipped_doc_ids"]
        repr_query_texts = batch["repr_query_text"]
        pos_texts = batch["pos_text"]
        pos_titles = batch["pos_title"]

        batch_no = self._batches_seen + 1

        # Collect all neighbor doc_ints for bulk text fetch
        all_neighbor_ints = set()
        neighbor_refs = 0
        docs_with_neighbors = 0
        docs_without_neighbors = 0
        for neighbors in neighbors_list:
            if neighbors is None or len(neighbors) == 0:
                docs_without_neighbors += 1
                continue
            docs_with_neighbors += 1
            neighbor_refs += len(neighbors)
            for neg in neighbors:
                all_neighbor_ints.add(int(neg["doc_int"]))

        should_log_batch = (
            self._log_every_batches > 0
            and (
                batch_no == 1
                or batch_no % self._log_every_batches == 0
            )
        )
        if should_log_batch:
            _print_telemetry(
                "DataJoiner batch start: shard=%s pid=%d batch=%d input_docs=%d "
                "docs_with_neighbors=%d docs_without_neighbors=%d neighbor_refs=%d "
                "unique_neighbor_doc_ints=%d duplicate_neighbor_refs=%d",
                self._shard_id if self._shard_id is not None else "?",
                self._pid,
                batch_no,
                len(doc_id_strs),
                docs_with_neighbors,
                docs_without_neighbors,
                neighbor_refs,
                len(all_neighbor_ints),
                neighbor_refs - len(all_neighbor_ints),
            )

        texts_lookup = {}
        lookup_t0 = time.time()
        if all_neighbor_ints:
            texts_lookup = self._doc_reader.get_texts_bulk(
                all_neighbor_ints, ["text", "title"]
            )
        lookup_s = time.time() - lookup_t0
        if should_log_batch:
            _print_telemetry(
                "DataJoiner text lookup done: shard=%s pid=%d batch=%d "
                "requested_unique_neighbors=%d returned_text_records=%d lookup_s=%.2f",
                self._shard_id if self._shard_id is not None else "?",
                self._pid,
                batch_no,
                len(all_neighbor_ints),
                len(texts_lookup),
                lookup_s,
            )

        out = {
            "doc_id_int": [],
            "all_qids": [],
            "all_query_texts": [],
            "skipped_doc_ids": [],
            "repr_query_text": [],
            "candidate_text": [],
            "pair_type": [],
            "neg_doc_int": [],
            "neg_embed_score": [],
        }

        input_docs = len(doc_id_strs)
        docs_with_queries = 0
        docs_without_queries = 0
        positive_pairs = 0
        negative_pairs = 0
        missing_neighbor_texts = 0
        empty_candidate_texts = 0
        build_t0 = time.time()
        for i in range(input_docs):
            doc_t0 = time.time()
            pairs_before_doc = len(out["doc_id_int"])
            did_int = doc_id_ints[i]
            did_str = doc_id_strs[i]
            neighbors = neighbors_list[i] if neighbors_list[i] is not None else []
            if should_log_batch:
                _print_telemetry(
                    "DataJoiner doc start: shard=%s pid=%d batch=%d doc_index=%d/%d "
                    "doc_id_int=%s doc_id_str=%s neighbor_count=%d",
                    self._shard_id if self._shard_id is not None else "?",
                    self._pid,
                    batch_no,
                    i + 1,
                    input_docs,
                    did_int,
                    did_str,
                    len(neighbors),
                )

            queries = (
                [str(q) for q in all_query_texts_list[i]]
                if all_query_texts_list[i] is not None
                else []
            )
            qids = (
                [int(q) for q in all_qids_list[i]]
                if all_qids_list[i] is not None
                else []
            )
            if len(queries) == 0:
                docs_without_queries += 1
                if should_log_batch:
                    _print_telemetry(
                        "DataJoiner doc skipped: shard=%s pid=%d batch=%d "
                        "doc_index=%d/%d doc_id_int=%s reason=no_queries "
                        "neighbor_count=%d elapsed_s=%.3f",
                        self._shard_id if self._shard_id is not None else "?",
                        self._pid,
                        batch_no,
                        i + 1,
                        input_docs,
                        did_int,
                        len(neighbors),
                        time.time() - doc_t0,
                    )
                continue
            docs_with_queries += 1

            repr_query = repr_query_texts[i]
            pos_text = pos_texts[i] or ""
            pos_title = pos_titles[i] or ""

            all_qids_json = json.dumps(qids)
            all_qtexts_json = json.dumps(queries)
            raw_skipped = (
                skipped_doc_ids_list[i] if skipped_doc_ids_list[i] is not None else []
            )
            skipped_entries = [
                {"doc_id": int(s["doc_id"]), "score": float(s["score"])}
                for s in raw_skipped
            ]
            skipped_json = json.dumps(skipped_entries)

            # Positive pair
            pos_candidate = (pos_title + " " + _truncate_text(pos_text, max_words=500))[
                :3000
            ].strip()
            if not pos_candidate:
                empty_candidate_texts += 1
            out["doc_id_int"].append(did_int)
            out["all_qids"].append(all_qids_json)
            out["all_query_texts"].append(all_qtexts_json)
            out["skipped_doc_ids"].append(skipped_json)
            out["repr_query_text"].append(repr_query)
            out["candidate_text"].append(pos_candidate)
            out["pair_type"].append("pos")
            out["neg_doc_int"].append(-1)
            out["neg_embed_score"].append(0.0)
            positive_pairs += 1

            # Negative pairs
            doc_missing_neighbor_texts = 0
            doc_empty_candidate_texts = 1 if not pos_candidate else 0
            for neg in neighbors:
                di = int(neg["doc_int"])
                t = texts_lookup.get(di, {})
                if not t:
                    missing_neighbor_texts += 1
                    doc_missing_neighbor_texts += 1
                neg_text = (
                    t.get("title", "")
                    + " "
                    + _truncate_text(t.get("text", ""), max_words=500)
                )[:3000].strip()
                if not neg_text:
                    empty_candidate_texts += 1
                    doc_empty_candidate_texts += 1
                out["doc_id_int"].append(did_int)
                out["all_qids"].append(all_qids_json)
                out["all_query_texts"].append(all_qtexts_json)
                out["skipped_doc_ids"].append(skipped_json)
                out["repr_query_text"].append(repr_query)
                out["candidate_text"].append(neg_text)
                out["pair_type"].append("neg")
                out["neg_doc_int"].append(di)
                out["neg_embed_score"].append(1.0 - neg["distance"])
                negative_pairs += 1
            if should_log_batch:
                _print_telemetry(
                    "DataJoiner doc done: shard=%s pid=%d batch=%d doc_index=%d/%d "
                    "doc_id_int=%s query_count=%d qid_count=%d skipped_count=%d "
                    "neighbor_count=%d emitted_pairs=%d missing_neighbor_texts=%d "
                    "empty_candidate_texts=%d pos_chars=%d pos_candidate_chars=%d "
                    "repr_query_chars=%d elapsed_s=%.3f",
                    self._shard_id if self._shard_id is not None else "?",
                    self._pid,
                    batch_no,
                    i + 1,
                    input_docs,
                    did_int,
                    len(queries),
                    len(qids),
                    len(raw_skipped),
                    len(neighbors),
                    len(out["doc_id_int"]) - pairs_before_doc,
                    doc_missing_neighbor_texts,
                    doc_empty_candidate_texts,
                    len(pos_text),
                    len(pos_candidate),
                    len(repr_query or ""),
                    time.time() - doc_t0,
                )
        build_s = time.time() - build_t0

        self._batches_seen += 1
        self._docs_seen += input_docs
        self._pairs_emitted += len(out["doc_id_int"])
        self._docs_with_queries += docs_with_queries
        self._docs_without_queries += docs_without_queries
        self._neighbor_refs_seen += neighbor_refs
        self._unique_neighbors_requested += len(all_neighbor_ints)
        self._neighbor_texts_returned += len(texts_lookup)
        self._missing_neighbor_texts += missing_neighbor_texts
        self._empty_candidate_texts += empty_candidate_texts
        self._positive_pairs += positive_pairs
        self._negative_pairs += negative_pairs
        self._lookup_s += lookup_s
        self._build_s += build_s

        should_log = (
            self._log_every_batches > 0
            and (
                self._batches_seen == 1
                or self._batches_seen % self._log_every_batches == 0
            )
        )
        if should_log:
            elapsed = max(time.time() - self._t0, 1e-9)
            batch_elapsed = time.time() - batch_t0
            docs_per_s = self._docs_seen / elapsed
            pairs_per_s = self._pairs_emitted / elapsed
            batch_pairs = len(out["doc_id_int"])
            progress_msg = (
                "DataJoiner progress: shard=%s pid=%d batches=%d docs=%d "
                "pairs=%d docs/s=%.1f pairs/s=%.1f last_batch_docs=%d "
                "docs_with_queries=%d docs_without_queries=%d "
                "neighbor_refs=%d unique_neighbors=%d duplicate_neighbor_refs=%d "
                "lookup_records=%d missing_neighbor_texts=%d "
                "empty_candidate_texts=%d last_batch_pairs=%d pos_pairs=%d "
                "neg_pairs=%d pairs_per_doc_with_queries=%.2f lookup_s=%.2f "
                "build_s=%.2f last_batch_s=%.2f totals: docs_with_queries=%d "
                "docs_without_queries=%d neighbor_refs=%d unique_neighbors=%d "
                "lookup_records=%d missing_neighbor_texts=%d empty_candidate_texts=%d "
                "pos_pairs=%d neg_pairs=%d lookup_s=%.2f build_s=%.2f"
            ) % (
                self._shard_id if self._shard_id is not None else "?",
                self._pid,
                self._batches_seen,
                self._docs_seen,
                self._pairs_emitted,
                docs_per_s,
                pairs_per_s,
                input_docs,
                docs_with_queries,
                docs_without_queries,
                neighbor_refs,
                len(all_neighbor_ints),
                neighbor_refs - len(all_neighbor_ints),
                len(texts_lookup),
                missing_neighbor_texts,
                empty_candidate_texts,
                batch_pairs,
                positive_pairs,
                negative_pairs,
                batch_pairs / max(docs_with_queries, 1),
                lookup_s,
                build_s,
                batch_elapsed,
                self._docs_with_queries,
                self._docs_without_queries,
                self._neighbor_refs_seen,
                self._unique_neighbors_requested,
                self._neighbor_texts_returned,
                self._missing_neighbor_texts,
                self._empty_candidate_texts,
                self._positive_pairs,
                self._negative_pairs,
                self._lookup_s,
                self._build_s,
            )
            _print_telemetry(progress_msg)

        return out


# ---------------------------------------------------------------------------
# Score preprocess / postprocess
# ---------------------------------------------------------------------------


def make_score_preprocess(
    reranker_model: str, store_raw_reranker_scores: bool = False
):
    """Build preprocess function for the score processor."""

    def preprocess(row: dict) -> dict:
        request_kwargs = {
            "model": reranker_model,
            "text_1": row["repr_query_text"],
            "text_2": row["candidate_text"],
        }
        if store_raw_reranker_scores:
            request_kwargs["use_activation"] = False

        return {
            "method": "score",
            "dtype": "ScoreRequest",
            "request_kwargs": request_kwargs,
            # carry-through columns
            "doc_id_int": row["doc_id_int"],
            "all_qids": row["all_qids"],
            "all_query_texts": row["all_query_texts"],
            "skipped_doc_ids": row["skipped_doc_ids"],
            "pair_type": row["pair_type"],
            "neg_doc_int": row["neg_doc_int"],
            "neg_embed_score": row["neg_embed_score"],
        }

    return preprocess


def _sigmoid(score: float) -> float:
    """Map a raw logit to a normalized score."""
    if score >= 0:
        z = math.exp(-score)
        return 1.0 / (1.0 + z)
    z = math.exp(score)
    return z / (1.0 + z)


def make_score_postprocess(store_raw_reranker_scores: bool = False):
    """Build postprocess function for the score processor."""

    def postprocess(row: dict) -> dict:
        # Extract score from ScoreResponse model_dump
        score = 0.0
        data = row.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                score = float(first.get("score", 0.0))
        raw_score = score
        if store_raw_reranker_scores:
            score = _sigmoid(raw_score)

        out = {
            "doc_id_int": row["doc_id_int"],
            "all_qids": row["all_qids"],
            "all_query_texts": row["all_query_texts"],
            "skipped_doc_ids": row["skipped_doc_ids"],
            "pair_type": row["pair_type"],
            "neg_doc_int": row["neg_doc_int"],
            "neg_embed_score": row["neg_embed_score"],
            "reranker_score": score,
        }
        if store_raw_reranker_scores:
            out["reranker_raw_score"] = raw_score
        return out

    return postprocess


# ---------------------------------------------------------------------------
# Aggregator — reassemble per-document output
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("doc_id_int", pa.int64()),
        pa.field("qids", pa.list_(pa.int64())),
        pa.field("query_texts", pa.list_(pa.string())),
        pa.field("pos_reranker_score", pa.float64()),
        pa.field(
            "neg_embed",
            pa.list_(
                pa.struct(
                    [
                        pa.field("doc_id", pa.int64()),
                        pa.field("score", pa.float64()),
                    ]
                )
            ),
        ),
        pa.field(
            "neg_reranker",
            pa.list_(
                pa.struct(
                    [
                        pa.field("doc_id", pa.int64()),
                        pa.field("score", pa.float64()),
                    ]
                )
            ),
        ),
        pa.field(
            "skipped_doc_ids",
            pa.list_(
                pa.struct(
                    [
                        pa.field("doc_id", pa.int64()),
                        pa.field("score", pa.float64()),
                    ]
                )
            ),
        ),
    ]
)


def make_aggregate_doc_group(store_raw_reranker_scores: bool = False):
    """Build aggregator for scored pairs grouped by doc_id_int."""

    def aggregate_doc_group(group: dict) -> dict:
        """Aggregate scored pairs for a single doc_id_int group.

        Used with ds.groupby("doc_id_int").map_groups().
        """
        pair_types = group["pair_type"]
        neg_doc_ints = group["neg_doc_int"]
        neg_embed_scores = group["neg_embed_score"]
        reranker_scores = group["reranker_score"]
        reranker_raw_scores = (
            group["reranker_raw_score"] if store_raw_reranker_scores else None
        )

        # All rows in this group share the same doc_id_int, all_qids, all_query_texts
        doc_id_int = group["doc_id_int"][0]
        all_qids_json = group["all_qids"][0]
        all_qtexts_json = group["all_query_texts"][0]

        pos_reranker_score = 0.0
        pos_reranker_raw_score = 0.0
        neg_embed = []
        neg_reranker = []
        neg_reranker_raw = []

        for i in range(len(pair_types)):
            if pair_types[i] == "pos":
                pos_reranker_score = float(reranker_scores[i])
                if store_raw_reranker_scores:
                    pos_reranker_raw_score = float(reranker_raw_scores[i])
            else:
                neg_embed.append(
                    {
                        "doc_id": int(neg_doc_ints[i]),
                        "score": float(neg_embed_scores[i]),
                    }
                )
                neg_reranker.append(
                    {"doc_id": int(neg_doc_ints[i]), "score": float(reranker_scores[i])}
                )
                if store_raw_reranker_scores:
                    neg_reranker_raw.append(
                        {
                            "doc_id": int(neg_doc_ints[i]),
                            "score": float(reranker_raw_scores[i]),
                        }
                    )

        skipped_json = group["skipped_doc_ids"][0]
        skipped = json.loads(skipped_json) if skipped_json else []

        out = {
            "doc_id_int": [doc_id_int],
            "qids": [json.loads(all_qids_json)],
            "query_texts": [json.loads(all_qtexts_json)],
            "pos_reranker_score": [pos_reranker_score],
            "neg_embed": [neg_embed],
            "neg_reranker": [neg_reranker],
            "skipped_doc_ids": [
                [
                    {"doc_id": int(s["doc_id"]), "score": float(s["score"])}
                    for s in skipped
                ]
            ],
        }
        if store_raw_reranker_scores:
            out["pos_reranker_raw_score"] = [pos_reranker_raw_score]
            out["neg_reranker_raw"] = [neg_reranker_raw]
        return out

    return aggregate_doc_group


# ---------------------------------------------------------------------------
# Pass 2 chunked execution helpers
# ---------------------------------------------------------------------------


def _build_pass2_dataset(
    ds, args, score_processor, aggregate_doc_group, shard_idx, joiner_batch_size=None
):
    """Attach the Pass 2 join/rerank/aggregate graph to an input dataset."""
    joiner_batch_size = (
        args.joiner_batch_size if joiner_batch_size is None else int(joiner_batch_size)
    )
    ds = ds.map_batches(
        DataJoiner,
        fn_constructor_kwargs={
            "index_dir": args.index_dir,
            "shard_id": shard_idx,
            "log_every_batches": args.joiner_log_every_batches,
            "doc_text_fetch_concurrency": args.doc_text_fetch_concurrency,
        },
        batch_size=joiner_batch_size,
        compute=ray.data.ActorPoolStrategy(size=args.joiner_concurrency),
        num_cpus=24,
    )
    ds = score_processor(ds)
    return ds.groupby("doc_id_int").map_groups(aggregate_doc_group)


def _pass2_shard_output_dir(args, shard_idx):
    return os.path.join(args.output_dir, f"shard_{shard_idx:05d}")


def _pass2_chunk_dir(shard_output_dir, chunk_idx):
    return os.path.join(shard_output_dir, f"chunk_{chunk_idx:05d}")


def _pass2_chunk_success_path(chunk_dir):
    return os.path.join(chunk_dir, "_SUCCESS")


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _pass2_chunk_meta_path(shard_output_dir):
    return os.path.join(shard_output_dir, "_chunked_meta.json")


def _pass2_materialized_marker_path(shard_output_dir):
    return os.path.join(shard_output_dir, "_MATERIALIZED_SUCCESS")


def _pass2_materialized_output_path(args, shard_output_dir):
    return os.path.join(shard_output_dir, args.materialized_output_basename)


def _is_pass2_materialized(args, shard_output_dir):
    return (
        os.path.exists(_pass2_materialized_marker_path(shard_output_dir))
        and os.path.isfile(_pass2_materialized_output_path(args, shard_output_dir))
    )


def _discover_completed_pass2_chunks(shard_output_dir):
    if not os.path.isdir(shard_output_dir):
        return set()
    completed = set()
    for name in os.listdir(shard_output_dir):
        if not name.startswith("chunk_"):
            continue
        chunk_dir = os.path.join(shard_output_dir, name)
        if not os.path.isdir(chunk_dir):
            continue
        if not os.path.exists(_pass2_chunk_success_path(chunk_dir)):
            continue
        try:
            completed.add(int(name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return completed


def _completed_pass2_chunks_from_ckpt(ckpt, shard_idx):
    raw = ckpt.setdefault("pass2_completed_chunks", {})
    return set(int(x) for x in raw.get(str(shard_idx), []))


def _mark_pass2_chunk_completed(ckpt, shard_idx, chunk_idx):
    raw = ckpt.setdefault("pass2_completed_chunks", {})
    key = str(shard_idx)
    chunks = raw.setdefault(key, [])
    _append_unique(chunks, int(chunk_idx))
    chunks.sort()


def _prepare_pass2_chunk_metadata(
    ckpt, shard_idx, shard_output_dir, chunk_size, num_rows
):
    num_chunks = math.ceil(num_rows / chunk_size) if num_rows else 0
    meta = {
        "shard_idx": int(shard_idx),
        "chunk_size": int(chunk_size),
        "num_rows": int(num_rows),
        "num_chunks": int(num_chunks),
    }

    ckpt_meta = ckpt.setdefault("pass2_chunk_meta", {})
    key = str(shard_idx)
    existing_ckpt_meta = ckpt_meta.get(key)
    existing_file_meta = _load_json_if_exists(_pass2_chunk_meta_path(shard_output_dir))
    existing_meta = existing_ckpt_meta or existing_file_meta

    if existing_meta is not None:
        comparable = {
            "shard_idx": int(existing_meta["shard_idx"]),
            "chunk_size": int(existing_meta["chunk_size"]),
            "num_rows": int(existing_meta["num_rows"]),
            "num_chunks": int(existing_meta["num_chunks"]),
        }
        if comparable != meta:
            raise ValueError(
                "Existing Pass 2 chunk metadata does not match this run for "
                f"shard {shard_idx}: existing={comparable}, current={meta}. "
                "Use the same --pass2-chunk-size or move the old chunk output/checkpoint."
            )

    ckpt_meta[key] = meta
    os.makedirs(shard_output_dir, exist_ok=True)
    _atomic_write_json(meta, _pass2_chunk_meta_path(shard_output_dir))
    return meta


def _ensure_chunked_output_dir_is_clean(shard_output_dir, shard_idx):
    if not os.path.isdir(shard_output_dir):
        return
    materialized_marker = _pass2_materialized_marker_path(shard_output_dir)
    flat_parquets = [
        name
        for name in os.listdir(shard_output_dir)
        if name.endswith(".parquet")
        and os.path.isfile(os.path.join(shard_output_dir, name))
    ]
    if flat_parquets:
        if os.path.exists(materialized_marker):
            return
        raise RuntimeError(
            f"Shard {shard_idx} output directory already has flat parquet files at "
            f"{shard_output_dir}. Move/delete it before using --pass2-chunk-size, "
            "or rerun without chunked Pass 2."
        )


def _run_pass2_chunked_shard(
    inter_path,
    shard_output_dir,
    args,
    score_processor,
    aggregate_doc_group,
    ckpt,
    ckpt_path,
    shard_idx,
    queryless_only=False,
):
    """Run Pass 2 for one shard in independently checkpointed chunks."""
    chunk_size = int(args.pass2_chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0 for chunked Pass 2")
    chunk_block_size = int(args.pass2_chunk_block_size)
    if chunk_block_size <= 0:
        raise ValueError("pass2_chunk_block_size must be > 0 for chunked Pass 2")

    _ensure_chunked_output_dir_is_clean(shard_output_dir, shard_idx)

    pf = pq.ParquetFile(inter_path)
    num_rows = pf.metadata.num_rows
    meta = _prepare_pass2_chunk_metadata(
        ckpt, shard_idx, shard_output_dir, chunk_size, num_rows
    )
    completed_chunks = _completed_pass2_chunks_from_ckpt(ckpt, shard_idx)
    discovered_chunks = _discover_completed_pass2_chunks(shard_output_dir)
    completed_chunks |= discovered_chunks
    for chunk_idx in discovered_chunks:
        _mark_pass2_chunk_completed(ckpt, shard_idx, chunk_idx)
    save_checkpoint(ckpt, ckpt_path)

    log.info(
        "  Chunked Pass 2 enabled for shard %d: rows=%d chunk_size=%d chunks=%d completed=%d",
        shard_idx,
        num_rows,
        chunk_size,
        meta["num_chunks"],
        len(completed_chunks),
    )

    if meta["num_chunks"] == 0:
        return True

    columns = ["all_query_texts"] if queryless_only else None
    for chunk_idx, batch in enumerate(pf.iter_batches(batch_size=chunk_size, columns=columns)):
        if chunk_idx in completed_chunks:
            log.info(
                "  Skipping completed Pass 2 chunk shard=%d chunk=%d/%d",
                shard_idx,
                chunk_idx + 1,
                meta["num_chunks"],
            )
            continue

        has_queries = any(batch.column("all_query_texts").to_pylist())
        if queryless_only and has_queries:
            continue

        chunk_dir = _pass2_chunk_dir(shard_output_dir, chunk_idx)
        if os.path.isdir(chunk_dir):
            shutil.rmtree(chunk_dir)

        chunk_rows = batch.num_rows
        log.info(
            "  Pass 2 chunk start: shard=%d chunk=%d/%d rows=%d -> %s",
            shard_idx,
            chunk_idx + 1,
            meta["num_chunks"],
            chunk_rows,
            chunk_dir,
        )
        chunk_t0 = time.time()
        if not has_queries:
            # Ray does not create a parquet directory for an empty dataset.
            # Keep an explicit empty parquet so materialization remains valid.
            schema = OUTPUT_SCHEMA
            if args.store_raw_reranker_scores:
                schema = schema.append(pa.field("pos_reranker_raw_score", pa.float64()))
                schema = schema.append(pa.field("neg_reranker_raw", schema.field("neg_reranker").type))
            os.makedirs(chunk_dir, exist_ok=True)
            pq.write_table(pa.Table.from_batches([], schema=schema),
                           os.path.join(chunk_dir, "part-empty.parquet"))
            _atomic_write_json(
                {"shard_idx": int(shard_idx), "chunk_idx": int(chunk_idx),
                 "chunk_rows": int(chunk_rows), "output_rows": 0,
                 "reason": "no_queries", "elapsed_s": time.time() - chunk_t0},
                _pass2_chunk_success_path(chunk_dir),
            )
            _mark_pass2_chunk_completed(ckpt, shard_idx, chunk_idx)
            save_checkpoint(ckpt, ckpt_path)
            completed_chunks.add(chunk_idx)
            log.info("  Completed queryless Pass 2 chunk: shard=%d chunk=%d/%d rows=%d",
                     shard_idx, chunk_idx + 1, meta["num_chunks"], chunk_rows)
            continue

        chunk_table = pa.Table.from_batches([batch])
        chunk_blocks = [
            pa.Table.from_batches([block])
            for block in chunk_table.to_batches(max_chunksize=chunk_block_size)
        ]
        log.info(
            "  Pass 2 chunk Ray input: shard=%d chunk=%d/%d blocks=%d "
            "block_size=%d joiner_batch_size=%d arrow_size=%s",
            shard_idx,
            chunk_idx + 1,
            meta["num_chunks"],
            len(chunk_blocks),
            chunk_block_size,
            args.joiner_batch_size,
            _format_bytes(chunk_table.nbytes),
        )
        ds = ray.data.from_arrow(chunk_blocks)
        ds = _build_pass2_dataset(
            ds,
            args,
            score_processor,
            aggregate_doc_group,
            shard_idx,
            joiner_batch_size=args.joiner_batch_size,
        )
        ds.write_parquet(chunk_dir)
        part_paths = [os.path.join(chunk_dir, name) for name in os.listdir(chunk_dir)
                      if name.endswith(".parquet")] if os.path.isdir(chunk_dir) else []
        if not part_paths or not sum(pq.read_metadata(path).num_rows for path in part_paths):
            raise RuntimeError(
                f"Pass 2 shard {shard_idx} chunk {chunk_idx} had queries but produced "
                "no output rows. Check the joiner/reranker errors before resuming."
            )
        _atomic_write_json(
            {
                "shard_idx": int(shard_idx),
                "chunk_idx": int(chunk_idx),
                "chunk_rows": int(chunk_rows),
                "elapsed_s": time.time() - chunk_t0,
            },
            _pass2_chunk_success_path(chunk_dir),
        )

        _mark_pass2_chunk_completed(ckpt, shard_idx, chunk_idx)
        save_checkpoint(ckpt, ckpt_path)
        completed_chunks.add(chunk_idx)
        log.info(
            "  Pass 2 chunk done: shard=%d chunk=%d/%d rows=%d elapsed=%.1fs checkpoint=%s",
            shard_idx,
            chunk_idx + 1,
            meta["num_chunks"],
            chunk_rows,
            time.time() - chunk_t0,
            ckpt_path,
        )

        del ds, chunk_blocks, chunk_table, batch
        gc.collect()

    return len(completed_chunks) >= meta["num_chunks"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    global ray, torch, serve, ServeDeploymentProcessorConfig, build_processor
    global ScoreRequest, LLMConfig, ModelLoadingConfig, build_llm_deployment
    import ray
    import ray.data
    import torch
    from ray import serve
    from ray.data.llm import ServeDeploymentProcessorConfig, build_processor
    from ray.llm._internal.serve.core.configs.openai_api_models import ScoreRequest
    from ray.serve.llm import LLMConfig, ModelLoadingConfig, build_llm_deployment


    if args.debug:
        args.log_level = "DEBUG"
        log.info("Debug mode enabled: logging=DEBUG, limiting to first 2 shards")
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Resolve rank / world-size ─────────────────────────────────────────
    if args.rank is None:
        args.rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if args.world_size is None:
        args.world_size = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    log.info("Rank %d / %d", args.rank, args.world_size)

    num_gpus = args.num_gpus or _detect_num_gpus()
    log.info("Detected %d GPUs", num_gpus)

    # ── List embedding shards and compute doc_id_int offsets ──────────────
    shard_paths = list_embedding_shards(args.embeddings_dir)
    shard_offsets = compute_shard_offsets(args.embeddings_dir, shard_paths)
    total_docs = shard_offsets[-1]
    log.info(
        "Embedding shards: %d shards, %d total documents", len(shard_paths), total_docs
    )

    # ── Compute query offsets ─────────────────────────────────────────────
    query_offsets = compute_query_offsets(args.embeddings_dir, shard_paths)
    total_queries = query_offsets[-1]
    log.info(
        "Query offsets: %d total queries across %d shards",
        total_queries,
        len(shard_paths),
    )

    # ── Load USearch index (memory-mapped) ────────────────────────────────
    index_path = os.path.join(args.index_dir, "index.usearch")
    log.info("Loading USearch index from %s (memory-mapped) ...", index_path)
    t0 = time.time()
    meta = Index.metadata(index_path)
    if meta is None:
        raise ValueError(f"Failed to read USearch index metadata from {index_path}")
    usearch_index = Index(
        ndim=meta["dimensions"],
        dtype=meta["kind_scalar"],
        metric=meta["kind_metric"],
    )
    _load_pbar = tqdm(total=None, desc="Loading USearch index", unit="vec")

    def _load_progress(processed: int, total: int) -> bool:
        if _load_pbar.total != total:
            _load_pbar.total = total
        _load_pbar.update(processed - _load_pbar.n)
        return True

    # usearch_index.load(index_path, progress=_load_progress)
    usearch_index.view(index_path, progress=_load_progress)
    _load_pbar.close()
    # usearch_index = Index.restore(index_path, view=False)
    num_docs = len(usearch_index)
    log.info(
        "USearch index loaded: %d vectors, %d dims (%.1fs)",
        num_docs,
        usearch_index.ndim,
        time.time() - t0,
    )

    if num_docs != total_docs:
        log.warning(
            "Mismatch: USearch index has %d docs but embedding shards have %d.",
            num_docs,
            total_docs,
        )

    # ── Assign shards to this rank ────────────────────────────────────────
    my_shards = [i for i in range(len(shard_paths)) if i % args.world_size == args.rank]
    if args.debug and len(my_shards) > 2:
        log.info("Debug mode: truncating from %d shards to 2", len(my_shards))
        my_shards = my_shards[:2]
    log.info("Rank %d owns %d / %d shards", args.rank, len(my_shards), len(shard_paths))

    # ── Checkpoint ────────────────────────────────────────────────────────
    ckpt_path = _checkpoint_path(args)
    ckpt = load_checkpoint(ckpt_path)
    ckpt.setdefault("pass1_completed", [])
    ckpt.setdefault("pass2_completed", [])
    pass1_done = set(ckpt.get("pass1_completed", []))
    pass2_done = set(ckpt.get("pass2_completed", []))

    materialized_done = []
    for si in my_shards:
        shard_output_dir = _pass2_shard_output_dir(args, si)
        if si not in pass2_done and _is_pass2_materialized(args, shard_output_dir):
            materialized_done.append(si)
            _append_unique(ckpt["pass2_completed"], si)
            pass2_done.add(si)
    if materialized_done:
        materialized_done.sort()
        save_checkpoint(ckpt, ckpt_path)
        log.info(
            "Marked %d materialized shard outputs as Pass 2 complete: %s",
            len(materialized_done),
            materialized_done,
        )

    if all(s in pass2_done for s in my_shards):
        log.info("All shards already completed for rank %d. Nothing to do.", args.rank)
        return

    # ── Intermediate storage ──────────────────────────────────────────────
    intermediate_dir = os.path.join(args.output_dir, "_intermediates")
    os.makedirs(intermediate_dir, exist_ok=True)

    def _inter_path(shard_idx):
        return os.path.join(
            intermediate_dir, f"shard_{shard_idx:05d}_rank{args.rank:04d}.parquet"
        )

    # ══════════════════════════════════════════════════════════════════════
    # PASS 1 — Document-centric embedding retrieval
    # ══════════════════════════════════════════════════════════════════════
    pending_p1 = [s for s in my_shards if s not in pass1_done]

    if pending_p1:
        log.info("═══ PASS 1: Embedding retrieval (%d shards) ═══", len(pending_p1))
        pass1_t0 = time.time()

        for li, si in enumerate(pending_p1):
            elapsed = time.time() - pass1_t0
            if li > 0:
                eta = elapsed / li * (len(pending_p1) - li)
                h, r = divmod(int(eta), 3600)
                m, s = divmod(r, 60)
                eta_str = f"{h:02d}:{m:02d}:{s:02d}"
            else:
                eta_str = "??:??:??"

            log.info(
                "═══ Pass 1 — Shard %d (%d/%d on rank %d) — ETA %s ═══",
                si,
                li + 1,
                len(pending_p1),
                args.rank,
                eta_str,
            )

            doc_id_strs, embeddings, query_payload = read_canonical_shard(
                shard_paths[si]
            )

            if args.debug and len(doc_id_strs) > 1000:
                log.info(
                    "Debug mode: truncating shard from %d to 1000 docs",
                    len(doc_id_strs),
                )
                doc_id_strs = doc_id_strs[:1000]
                embeddings = embeddings[:1000]
                for k in query_payload:
                    query_payload[k] = query_payload[k][:1000]

            offset = shard_offsets[si]
            qid_base = query_offsets[si]
            doc_id_ints = list(range(offset, offset + len(doc_id_strs)))
            log.info(
                "  Read %d embeddings (doc_id_int %d..%d)",
                len(doc_id_strs),
                offset,
                offset + len(doc_id_strs) - 1,
            )

            t0 = time.time()
            neighbors, skipped = phase1_retrieve(
                embeddings,
                doc_id_ints,
                usearch_index,
                args.topk,
                num_docs,
                args.search_threads,
                skip_top_n=args.skip_top_n,
            )
            log.info("  USearch search done in %.1fs", time.time() - t0)

            running_qid = 0
            intermediate = []
            for i in range(len(doc_id_strs)):
                n_queries = len(query_payload["all_query_texts"][i])
                qids = list(
                    range(qid_base + running_qid, qid_base + running_qid + n_queries)
                )
                running_qid += n_queries

                intermediate.append(
                    {
                        "doc_id_str": doc_id_strs[i],
                        "doc_id_int": doc_id_ints[i],
                        "neighbors": [
                            {"doc_int": di, "distance": d} for di, d in neighbors[i]
                        ],
                        "skipped_doc_ids": [
                            {"doc_id": di, "score": 1.0 - d} for di, d in skipped[i]
                        ],
                        "all_query_texts": query_payload["all_query_texts"][i],
                        "all_qids": qids,
                        "repr_query_text": query_payload["repr_query_text"][i],
                        "pos_text": query_payload["pos_text"][i],
                        "pos_title": query_payload["pos_title"][i],
                    }
                )

            write_intermediate_parquet(intermediate, _inter_path(si))

            ckpt["pass1_completed"].append(si)
            save_checkpoint(ckpt, ckpt_path)
            pass1_done.add(si)

            del doc_id_strs, embeddings, query_payload, neighbors, intermediate
            gc.collect()

        log.info("═══ PASS 1 complete. ═══")
    else:
        log.info("Pass 1 already complete for rank %d.", args.rank)

    # Free USearch index before GPU-heavy Pass 2
    del usearch_index
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # PASS 2 — Ray Data pipeline: join + rerank + aggregate
    # ══════════════════════════════════════════════════════════════════════
    pending_p2 = [s for s in my_shards if s not in pass2_done]

    # Finish chunks with no query payload before allocating Ray/LLM replicas.
    if args.pass2_chunk_size > 0:
        for si in pending_p2:
            inter_path = _inter_path(si)
            if not os.path.isfile(inter_path):
                continue
            if _run_pass2_chunked_shard(
                inter_path, _pass2_shard_output_dir(args, si), args,
                None, None, ckpt, ckpt_path, si, queryless_only=True,
            ):
                _append_unique(ckpt["pass2_completed"], si)
                save_checkpoint(ckpt, ckpt_path)
                pass2_done.add(si)
        pending_p2 = [s for s in pending_p2 if s not in pass2_done]

    if not pending_p2:
        log.info("Pass 2 already complete for rank %d.", args.rank)
        return

    log.info("═══ PASS 2: Ray Data reranking pipeline (%d shards) ═══", len(pending_p2))

    # ── Pre-convert doc_texts to Arrow IPC (before Ray workers need them) ─
    preconvert_doc_texts_to_arrow(args.index_dir)

    # ── Ray init ──────────────────────────────────────────────────────────
    ray_address = os.environ.pop("RAY_ADDRESS", None)
    if ray_address:
        log.warning(
            "RAY_ADDRESS was set to %s but will be ignored; starting a local cluster.",
            ray_address,
        )

    ray_dashboard_port = int(os.getenv("RAY_DASHBOARD_PORT", "8265"))
    ray_metrics_export_port = int(os.getenv("RAY_METRICS_EXPORT_PORT", "9090"))
    ray_init_kwargs = dict(
        address="local",
        ignore_reinit_error=True,
        _metrics_export_port=ray_metrics_export_port,
        dashboard_host="127.0.0.1",
        dashboard_port=ray_dashboard_port,
        # num_gpus=num_gpus,
        logging_level=logging.DEBUG,
    )
    if args.ray_object_store_memory_gb is not None:
        ray_init_kwargs["object_store_memory"] = int(
            args.ray_object_store_memory_gb * (1024**3)
        )
    log.info(
        "Ray init config: object_store_memory_gb=%s, dashboard_port=%s, metrics_port=%s",
        args.ray_object_store_memory_gb,
        ray_dashboard_port,
        ray_metrics_export_port,
    )

    ray.init(**ray_init_kwargs)
    resources = ray.cluster_resources()
    log.info("Ray initialised: %s", resources)

    # ── Deploy reranker via Ray Serve LLM ─────────────────────────────────
    reranker_app_name = f"hard-neg-reranker-app-r{args.rank}"
    reranker_deployment_name = f"hard-neg-reranker-deployment-r{args.rank}"

    extra_engine_kwargs = {}
    if "gemma" in args.reranker_model.lower():
        extra_engine_kwargs["block_size"] = 64
    if args.reranker_chat_template:
        extra_engine_kwargs["chat_template"] = args.reranker_chat_template

    engine_kwargs = {
        "max_model_len": args.reranker_max_model_len,
        "runner": "pooling",
        "gpu_memory_utilization": args.reranker_gpu_memory_utilization,
        "tensor_parallel_size": args.reranker_tp_size,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": args.reranker_max_num_batched_tokens,
        "max_num_seqs": args.reranker_max_num_seqs,
        "enforce_eager": args.enforce_eager,
        "dtype": args.dtype,
        "disable_log_stats": False,
        **extra_engine_kwargs,
    }
    if args.reranker_cudagraph_capture_sizes is not None:
        engine_kwargs["cudagraph_capture_sizes"] = args.reranker_cudagraph_capture_sizes
    if args.reranker_max_cudagraph_capture_size is not None:
        engine_kwargs["max_cudagraph_capture_size"] = (
            args.reranker_max_cudagraph_capture_size
        )

    log.info(
        "Reranker cudagraph config: enforce_eager=%s, capture_sizes=%s, max_capture_size=%s",
        args.enforce_eager,
        args.reranker_cudagraph_capture_sizes,
        args.reranker_max_cudagraph_capture_size,
    )

    reranker_llm_config = LLMConfig(
        model_loading_config=ModelLoadingConfig(
            model_id=args.reranker_model,
            model_source=args.reranker_model,
        ),
        deployment_config=dict(
            name=reranker_deployment_name,
            autoscaling_config=dict(
                min_replicas=args.reranker_concurrency,
                max_replicas=args.reranker_concurrency,
            ),
        ),
        engine_kwargs=engine_kwargs,
        placement_group_config={
            "bundles": [{"GPU": 1, "CPU": 4}] * args.reranker_tp_size,
            "strategy": "PACK",
        },
        runtime_env=dict(
            env_vars={
                "VLLM_DISABLE_COMPILE_CACHE": "1",
                "VLLM_LOGGING_LEVEL": "INFO",

            },
        ),
    )

    reranker_app = build_llm_deployment(
        reranker_llm_config,
        name_prefix=f"hard-neg-reranker-r{args.rank}",
        override_serve_options={"name": reranker_deployment_name},
    )
    serve.run(
        reranker_app,
        name=reranker_app_name,
        route_prefix=f"/{reranker_deployment_name}",
        logging_config={"enable_access_log": False},
    )
    log.info("Reranker Serve app deployed: %s", reranker_app_name)

    # ── Build score processor ─────────────────────────────────────────────
    score_processor_config = ServeDeploymentProcessorConfig(
        deployment_name=reranker_deployment_name,
        app_name=reranker_app_name,
        dtype_mapping={"ScoreRequest": ScoreRequest},
        concurrency=args.reranker_concurrency * 4,
        batch_size=args.reranker_batch_size,
        should_continue_on_error=True,
    )

    score_processor = build_processor(
        score_processor_config,
        preprocess=make_score_preprocess(
            args.reranker_model, args.store_raw_reranker_scores
        ),
        postprocess=make_score_postprocess(args.store_raw_reranker_scores),
    )
    aggregate_doc_group = make_aggregate_doc_group(args.store_raw_reranker_scores)

    # ── Process each shard through the Ray Data pipeline ──────────────────
    pass2_t0 = time.time()
    for li, si in enumerate(pending_p2):
        elapsed = time.time() - pass2_t0
        if li > 0:
            eta = elapsed / li * (len(pending_p2) - li)
            h, r = divmod(int(eta), 3600)
            m, s = divmod(r, 60)
            eta_str = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            eta_str = "??:??:??"

        log.info(
            "═══ Pass 2 — Shard %d (%d/%d on rank %d) — ETA %s ═══",
            si,
            li + 1,
            len(pending_p2),
            args.rank,
            eta_str,
        )

        inter_path = _inter_path(si)
        if not os.path.exists(inter_path):
            log.warning(
                "Intermediate file missing: %s — skipping shard %d", inter_path, si
            )
            continue

        shard_output_dir = _pass2_shard_output_dir(args, si)
        if _is_pass2_materialized(args, shard_output_dir):
            log.info(
                "  Shard %d is already materialized at %s; marking Pass 2 complete.",
                si,
                _pass2_materialized_output_path(args, shard_output_dir),
            )
            _append_unique(ckpt["pass2_completed"], si)
            save_checkpoint(ckpt, ckpt_path)
            pass2_done.add(si)
            continue

        if args.pass2_chunk_size > 0:
            shard_complete = _run_pass2_chunked_shard(
                inter_path,
                shard_output_dir,
                args,
                score_processor,
                aggregate_doc_group,
                ckpt,
                ckpt_path,
                si,
            )
            if not shard_complete:
                log.warning("Shard %d did not complete all Pass 2 chunks", si)
                continue
            log.info("  Wrote chunked shard output -> %s", shard_output_dir)
        else:
            # Read intermediate parquet as Ray Dataset
            ds = ray.data.read_parquet(inter_path)
            ds = _build_pass2_dataset(
                ds, args, score_processor, aggregate_doc_group, si
            )

            # Write output (Ray Data writes to a directory; one subdir per shard)
            ds.write_parquet(shard_output_dir)
            log.info("  Wrote shard output -> %s", shard_output_dir)

        # Checkpoint
        _append_unique(ckpt["pass2_completed"], si)
        save_checkpoint(ckpt, ckpt_path)
        pass2_done.add(si)

        # Clean up intermediate
        try:
            os.remove(inter_path)
        except OSError:
            pass

    log.info("═══ PASS 2 complete. ═══")

    # Clean up Serve app
    log.info("Deleting reranker Serve app %s ...", reranker_app_name)
    serve.delete(reranker_app_name, _blocking=True)
    log.info("Reranker Serve app deleted.")

    log.info(
        "Pipeline complete. Rank %d/%d processed %d shards -> %s",
        args.rank,
        args.world_size,
        len(my_shards),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
