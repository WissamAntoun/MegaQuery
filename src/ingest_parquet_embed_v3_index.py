#!/usr/bin/env python3
"""
Ingest pipeline v3 — Step 2: Verify completeness and build USearch index.

Reads all canonical shard parquets produced by step 1 (shard workers),
verifies that every input file was processed, then builds the final USearch
index and writes doc_id_map + doc_texts.

Canonical shard schema:
  doc_id_local (int64), doc_id_str (string), pos_text (string),
  pos_title (string), all_query_texts (list<string>),
  repr_query_text (string), embedding (fixed_size_list<float32>)

Verification checks:
  1. All expected shard manifests are present (0..world_size-1).
  2. The union of input files across all shards covers every file matched
     by --glob (no file missed, no file processed twice).
  3. Each shard parquet has the expected number of rows per its manifest.
  4. Embedding dimensions are consistent across all shards.

Outputs (in --output-dir):
  index.usearch              — USearch HNSW index keyed by global doc_id_int
  doc_id_map.parquet         — Mapping: doc_id_int (int64) <-> doc_id_str (string)
  doc_texts/shard_XXXXX.parquet — Sharded document texts (text, title)
  doc_texts/meta.json        — Shard metadata
  verification.json          — Verification report

Usage:
    python src/ingest_parquet_embed_v3_index.py \
        --glob "/data/shard-*.parquet" \
        --output-dir /output/embed_index_v3 \
        --world-size 8
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

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="v3 index builder: verify embeddings and build USearch index"
    )

    g = p.add_argument_group("Input data (for verification)")
    g.add_argument("--glob", required=True,
                   help="Glob pattern for original input parquet files (same as step 1)")

    g = p.add_argument_group("Embeddings")
    g.add_argument("--output-dir", required=True,
                   help="Base output directory (same as step 1)")
    g.add_argument("--world-size", type=int, required=True,
                   help="Total number of shard workers from step 1")

    g = p.add_argument_group("USearch index")
    g.add_argument("--metric", default="cos",
                   help="Distance metric: cos, ip, l2sq [default: cos]")
    g.add_argument("--dtype", default="f16",
                   help="Quantization: f32, f16, i8 [default: f16]")
    g.add_argument("--connectivity", type=int, default=16,
                   help="HNSW graph connectivity [default: 16]")
    g.add_argument("--expansion-add", type=int, default=128,
                   help="HNSW expansion at add time [default: 128]")
    g.add_argument("--expansion-search", type=int, default=64,
                   help="HNSW expansion at search time [default: 64]")

    g = p.add_argument_group("Output control")
    g.add_argument("--doc-texts-shard-size", type=int, default=500_000,
                   help="Rows per sharded doc_texts parquet file [default: 500000]")
    g.add_argument("--index-batch-size", type=int, default=100_000,
                   help="Vectors to add to USearch at once [default: 100000]")

    g.add_argument("--log-level", default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_shards(embed_dir, world_size, expected_files):
    """Verify all shard manifests are present and cover all input files.

    Returns (manifests_list, verification_report_dict).
    Raises RuntimeError on verification failure.
    """
    report = {
        "world_size": world_size,
        "expected_input_files": len(expected_files),
        "shards": {},
        "status": "pending",
    }
    expected_abs = set(os.path.abspath(f) for f in expected_files)

    manifests = []
    covered_files = set()
    total_docs = 0
    embedding_dims = set()
    missing_shards = []
    missing_parquets = []

    for rank in range(world_size):
        manifest_path = os.path.join(embed_dir, f"shard_{rank:05d}.manifest.json")
        parquet_path = os.path.join(embed_dir, f"shard_{rank:05d}.parquet")

        if not os.path.exists(manifest_path):
            missing_shards.append(rank)
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)
        manifests.append(manifest)

        if not os.path.exists(parquet_path):
            missing_parquets.append(rank)
            continue

        # Verify parquet row count
        pf_meta = pq.read_metadata(parquet_path)
        actual_rows = pf_meta.num_rows
        expected_rows = manifest["num_docs"]
        if actual_rows != expected_rows:
            raise RuntimeError(
                f"Shard {rank}: manifest says {expected_rows} docs but "
                f"parquet has {actual_rows} rows"
            )

        # Track coverage
        shard_files = set(manifest["input_files"])
        overlap = covered_files & shard_files
        if overlap:
            raise RuntimeError(
                f"Shard {rank}: files already covered by another shard: {overlap}"
            )
        covered_files |= shard_files
        total_docs += manifest["num_docs"]

        if manifest["num_docs"] > 0:
            embedding_dims.add(manifest["embedding_dim"])

        report["shards"][rank] = {
            "num_docs": manifest["num_docs"],
            "input_files": len(manifest["input_files"]),
            "encode_time_s": manifest.get("encode_time_s"),
            "docs_per_sec": manifest.get("docs_per_sec"),
        }

    # Check for errors
    errors = []
    if missing_shards:
        errors.append(f"Missing shard manifests: {missing_shards}")
    if missing_parquets:
        errors.append(f"Missing shard parquets: {missing_parquets}")

    uncovered = expected_abs - covered_files
    if uncovered:
        errors.append(
            f"{len(uncovered)} input files not covered by any shard: "
            f"{sorted(uncovered)[:5]}{'…' if len(uncovered) > 5 else ''}"
        )

    extra = covered_files - expected_abs
    if extra:
        log.warning(
            "%d files in shard manifests not in current --glob (may be OK if glob changed): %s",
            len(extra), sorted(extra)[:5],
        )

    if len(embedding_dims) > 1:
        errors.append(f"Inconsistent embedding dimensions across shards: {embedding_dims}")

    if errors:
        report["status"] = "FAILED"
        report["errors"] = errors
        raise RuntimeError(
            "Verification failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    report["status"] = "OK"
    report["total_docs"] = total_docs
    report["embedding_dim"] = embedding_dims.pop() if embedding_dims else 0

    log.info(
        "Verification OK: %d shards, %d total docs, %d input files covered, dim=%d",
        world_size, total_docs, len(covered_files),
        report["embedding_dim"],
    )
    return manifests, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    embed_dir = os.path.join(args.output_dir, "embeddings")
    if not os.path.isdir(embed_dir):
        raise FileNotFoundError(
            f"Embeddings directory not found: {embed_dir}. Run step 1 first."
        )

    # Discover expected input files
    expected_files = sorted(globlib.glob(args.glob, recursive=True))
    if not expected_files:
        raise FileNotFoundError(f"No files matched: {args.glob}")
    log.info("Expected input files: %d", len(expected_files))

    # -- Phase 1: Verify completeness ------------------------------------------
    log.info("Phase 1: Verifying shard completeness …")
    manifests, report = verify_shards(embed_dir, args.world_size, expected_files)

    verification_path = os.path.join(args.output_dir, "verification.json")
    with open(verification_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Verification report: %s", verification_path)

    total_docs = report["total_docs"]
    ndim = report["embedding_dim"]

    if total_docs == 0:
        log.warning("No documents to index. Exiting.")
        return

    # -- Phase 2: Build USearch index + write doc_id_map + doc_texts -----------
    log.info("Phase 2: Building index from %d shard parquets …", len(manifests))

    index_path = os.path.join(args.output_dir, "index.usearch")
    doc_id_map_path = os.path.join(args.output_dir, "doc_id_map.parquet")
    doc_texts_dir = os.path.join(args.output_dir, "doc_texts")
    os.makedirs(doc_texts_dir, exist_ok=True)

    # Initialize USearch index
    index = Index(
        ndim=ndim,
        metric=args.metric,
        dtype=args.dtype,
        connectivity=args.connectivity,
        expansion_add=args.expansion_add,
        expansion_search=args.expansion_search,
    )

    # Schemas
    map_schema = pa.schema([
        pa.field("doc_id_int", pa.int64()),
        pa.field("doc_id_str", pa.string()),
    ])
    texts_schema = pa.schema([
        pa.field("doc_id_int", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("title", pa.string()),
    ])

    # doc_id_map writer
    map_tmp = doc_id_map_path + ".tmp"
    map_writer = pq.ParquetWriter(map_tmp, map_schema)

    # doc_texts sharded writer state
    texts_shard_idx = 0
    texts_rows_in_shard = 0
    texts_writer = pq.ParquetWriter(
        os.path.join(doc_texts_dir, "shard_00000.parquet"), texts_schema
    )

    global_id = 0
    t0 = time.time()

    # Process shards in rank order for deterministic global IDs
    for rank in tqdm(range(args.world_size), desc="Processing shards", unit="shard"):
        parquet_path = os.path.join(embed_dir, f"shard_{rank:05d}.parquet")
        manifest_path = os.path.join(embed_dir, f"shard_{rank:05d}.manifest.json")

        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest["num_docs"] == 0:
            continue

        pf = pq.ParquetFile(parquet_path)

        for batch in pf.iter_batches(batch_size=args.index_batch_size):
            n = batch.num_rows

            # Extract embeddings (Arrow list column -> 2-D numpy array)
            # pc.list_flatten respects array offsets (unlike .values on sliced batches)
            emb_col = batch.column("embedding")
            embeddings = pc.list_flatten(emb_col).to_numpy(zero_copy_only=False).reshape(n, ndim).astype(np.float32)

            # Assign global IDs
            global_ids = np.arange(global_id, global_id + n, dtype=np.int64)

            # Add to USearch index
            index.add(global_ids, embeddings, log=False, threads=0)

            # Write doc_id_map batch
            doc_id_strs = batch.column("doc_id_str").to_pylist()
            map_batch = pa.RecordBatch.from_arrays(
                [pa.array(global_ids.tolist(), type=pa.int64()),
                 pa.array(doc_id_strs, type=pa.string())],
                schema=map_schema,
            )
            map_writer.write_batch(map_batch)

            # Write doc_texts batch — map canonical shard columns to output schema
            texts_dict = {
                "doc_id_int": global_ids.tolist(),
                "text": batch.column("pos_text").to_pylist(),
                "title": batch.column("pos_title").to_pylist(),
            }

            # Write doc_texts, splitting across shard boundaries so each
            # shard has exactly doc_texts_shard_size rows (except the last).
            written = 0
            while written < n:
                capacity = args.doc_texts_shard_size - texts_rows_in_shard
                chunk_size = min(capacity, n - written)

                chunk_dict = {}
                for col_name, col_vals in texts_dict.items():
                    chunk_dict[col_name] = col_vals[written:written + chunk_size]

                texts_writer.write_batch(
                    pa.RecordBatch.from_pydict(chunk_dict, schema=texts_schema)
                )
                texts_rows_in_shard += chunk_size
                written += chunk_size

                # Rotate when shard is exactly full
                if texts_rows_in_shard >= args.doc_texts_shard_size:
                    texts_writer.close()
                    texts_shard_idx += 1
                    texts_writer = pq.ParquetWriter(
                        os.path.join(
                            doc_texts_dir,
                            f"shard_{texts_shard_idx:05d}.parquet",
                        ),
                        texts_schema,
                    )
                    texts_rows_in_shard = 0

            global_id += n

            if global_id % 1_000_000 < args.index_batch_size:
                elapsed = time.time() - t0
                log.info(
                    "  %d / %d docs indexed (%.1fs, %.0f docs/s)",
                    global_id, total_docs, elapsed,
                    global_id / max(elapsed, 1e-9),
                )

    # Close writers
    map_writer.close()
    os.replace(map_tmp, doc_id_map_path)

    texts_writer.close()
    num_text_shards = texts_shard_idx + (1 if texts_rows_in_shard > 0 else 0)
    texts_meta = {"shard_size": args.doc_texts_shard_size, "num_shards": num_text_shards}
    with open(os.path.join(doc_texts_dir, "meta.json"), "w") as f:
        json.dump(texts_meta, f)

    # Save index
    log.info("Saving USearch index …")
    index.save(index_path)

    elapsed = time.time() - t0
    log.info("Done. %d documents indexed in %.1fs.", global_id, elapsed)
    log.info("  Index:      %s (%d vectors, dim=%d)", index_path, len(index), ndim)
    log.info("  Doc ID map: %s", doc_id_map_path)
    log.info("  Doc texts:  %s (%d shards)", doc_texts_dir, num_text_shards)
    log.info("  Verification: %s", verification_path)


if __name__ == "__main__":
    main()
