#!/usr/bin/env python3
"""
Compact synthetic generation parquet output.

The synth gen pipeline writes one row per (document, query) pair, duplicating
pos_text and pos_title ~4x per document.  This script:

  1. Reads all input parquet files matching a glob.
  2. Groups rows by document ID, aggregating qids and query texts into lists.
  3. Writes compacted parquets with one row per document.

Parallelism is achieved by splitting input files across workers (multiprocessing).
Each worker processes its files in batches of --files-per-batch, aggregates docs
within each batch, writes one output file per batch, then frees memory. This keeps
RAM usage bounded even with TBs of data.

Usage:
    python src/compact_synth_parquets.py \
        --glob "/data/synth_out/*.parquet" \
        --output-dir /data/synth_compacted \
        --query-id-field id \
        --num-workers 8 \
        --files-per-batch 50
"""

import argparse
import glob as globlib
import logging
import multiprocessing as mp
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compact synth gen parquets (deduplicate documents)"
    )

    p.add_argument("--glob", required=True, help="Glob pattern for input parquet files")
    p.add_argument(
        "--output-dir", required=True, help="Output directory for compacted parquets"
    )
    p.add_argument(
        "--query-id-field",
        default="id",
        help="Column containing generated variant IDs [default: id]",
    )
    p.add_argument(
        "--non-dup-fields",
        default="query_length_target,request_id,stage1_difficulty,stage1_question_type,time_taken,query,stage2_raw",
        help="Comma-separated fields aggregated into aligned per-query lists",
    )
    p.add_argument(
        "--to-remove-fields",
        default="__inference_error__,batch_uuid,prompt_logprobs,prompt_token_ids,service_tier,stage1_raw,system_fingerprint,__id__,persona_candidates",
        help="Comma-separated list of extra columns to remove from output (e.g. internal metadata) [default: __inference_error__,batch_uuid,prompt_logprobs,prompt_token_ids,service_tier,stage1_raw,system_fingerprint,__id__]",
    )
    p.add_argument(
        "--strip-doc-id-suffix",
        default="__s1d_",
        help="Suffix pattern to strip from doc IDs before grouping (e.g. '__s1d_'). "
        "Set to empty string to disable. [default: __s1d_]",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers [default: 8]",
    )
    p.add_argument(
        "--files-per-batch",
        type=int,
        default=50,
        help="Input files to aggregate before flushing to disk [default: 50]",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Row batch size when reading input parquets [default: 8192]",
    )

    return p.parse_args()


def _flush_batch(results, dup_fields, non_dup_fields, qid_field, out_path):
    """Write aggregated results to a single parquet file."""
    if not results:
        return 0, 0

    num_docs = len(results)
    total_queries = (
        sum(len(r["_non_dup"][non_dup_fields[0]]) for r in results)
        if non_dup_fields
        else 0
    )

    columns = {
        qid_field: pa.array([r[qid_field] for r in results], type=pa.string()),
    }
    for c in dup_fields:
        columns[c] = pa.array([r.get(c) for r in results])
    for c in non_dup_fields:
        columns[c] = pa.array([r["_non_dup"][c] for r in results])

    table = pa.table(columns)
    tmp_path = out_path + ".tmp"
    pq.write_table(table, tmp_path, row_group_size=100_000)
    os.replace(tmp_path, out_path)

    return num_docs, total_queries


def _process_files(
    files, qid_field, non_dup_fields, to_remove_fields, strip_suffix, batch_size
):
    """Read a list of parquet files and return (results, dup_fields).

    results: list of dicts, one per unique doc. Each dict has:
        - qid_field: the stripped doc id
        - dup_fields keys: duplicated columns (kept from first occurrence)
        - "_non_dup": dict of {field: [values...]} for per-query columns
    dup_fields: list of column names that are duplicated across rows.
    """
    doc_order = {}
    results = []
    dup_fields = None
    skip = set(to_remove_fields) | set(non_dup_fields) | {qid_field}

    for fpath in files:
        pf = pq.ParquetFile(fpath)

        if dup_fields is None:
            dup_fields = [c for c in pf.schema_arrow.names if c not in skip]

        for rb in pf.iter_batches(batch_size=batch_size):
            cols = {f: rb.column(f).to_pylist() for f in rb.schema.names}
            for i in range(rb.num_rows):
                raw_id = cols[qid_field][i]
                if strip_suffix and strip_suffix in str(raw_id):
                    doc_id = str(raw_id).rsplit(strip_suffix, 1)[0]
                else:
                    doc_id = str(raw_id)

                if doc_id not in doc_order:
                    idx = len(results)
                    doc_order[doc_id] = idx
                    entry = {
                        qid_field: doc_id,
                        "_non_dup": {c: [] for c in non_dup_fields},
                    }
                    for c in dup_fields:
                        if c in cols:
                            entry[c] = cols[c][i]
                    results.append(entry)

                entry = results[doc_order[doc_id]]
                for c in non_dup_fields:
                    # Keep null slots so query metadata remains positionally aligned.
                    entry["_non_dup"][c].append(cols[c][i] if c in cols else None)

    return results, dup_fields or []


def _worker(worker_args):
    """Process assigned files in batches, writing one output file per batch."""
    (
        worker_id,
        files,
        qid_field,
        non_dup_fields,
        to_remove_fields,
        strip_suffix,
        batch_size,
        files_per_batch,
        out_dir,
    ) = worker_args

    total_files = len(files)
    total_docs = 0
    total_queries = 0
    batch_idx = 0

    pbar = tqdm(
        total=total_files,
        desc=f"Worker {worker_id:02d}",
        unit="file",
        position=worker_id + 1,
        leave=False,
    )

    for start in range(0, total_files, files_per_batch):
        batch_files = files[start : start + files_per_batch]

        results, dup_fields = _process_files(
            batch_files,
            qid_field,
            non_dup_fields,
            to_remove_fields,
            strip_suffix,
            batch_size,
        )

        out_path = os.path.join(
            out_dir, f"compact_w{worker_id:03d}_b{batch_idx:04d}.parquet"
        )

        n_docs, n_queries = _flush_batch(
            results, dup_fields, non_dup_fields, qid_field, out_path
        )
        total_docs += n_docs
        total_queries += n_queries
        batch_idx += 1

        # Free memory
        del results, dup_fields

        pbar.update(len(batch_files))

    pbar.close()
    return worker_id, total_files, total_docs, total_queries


def compact(args):
    all_files = sorted(globlib.glob(args.glob, recursive=True))
    if not all_files:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    num_workers = min(args.num_workers, len(all_files))
    log.info(
        "Found %d input files, using %d workers, %d files/batch",
        len(all_files),
        num_workers,
        args.files_per_batch,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Split files across workers (round-robin for balanced sizes)
    file_chunks = [[] for _ in range(num_workers)]
    for i, f in enumerate(all_files):
        file_chunks[i % num_workers].append(f)

    worker_args = [
        (
            wid,
            file_chunks[wid],
            args.query_id_field,
            args.non_dup_fields.split(","),
            args.to_remove_fields.split(","),
            args.strip_doc_id_suffix,
            args.batch_size,
            args.files_per_batch,
            args.output_dir,
        )
        for wid in range(num_workers)
    ]

    t0 = time.time()
    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_worker, worker_args),
                total=num_workers,
                desc="Workers",
                unit="worker",
            )
        )

    total_files = sum(r[1] for r in results)
    total_docs = sum(r[2] for r in results)
    total_queries = sum(r[3] for r in results)
    log.info(
        "Done in %.1fs: %d docs, %d queries (from %d input files)",
        time.time() - t0,
        total_docs,
        total_queries,
        total_files,
    )


def main():
    args = parse_args()
    compact(args)


if __name__ == "__main__":
    main()
