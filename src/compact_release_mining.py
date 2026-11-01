#!/usr/bin/env python3
"""Compact mining inputs with Arrow in isolated, restartable worker processes."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import logging
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from release_data import sha256, write_json


LOG = logging.getLogger(__name__)
CACHE_VERSION = 1
SCORE_PAIRS = pa.list_(pa.struct([("doc_id", pa.int64()), ("score", pa.float64())]))
MINING_SCHEMA = pa.schema([
    ("doc_id_int", pa.int64()), ("qids", pa.list_(pa.int64())),
    ("query_texts", pa.list_(pa.string())),
    ("pos_reranker_score", pa.float64()), ("pos_reranker_raw_score", pa.float64()),
    ("neg_embed", SCORE_PAIRS), ("neg_reranker", SCORE_PAIRS),
    ("neg_reranker_raw", SCORE_PAIRS), ("skipped_doc_ids", SCORE_PAIRS),
])
OPTIONAL_COLUMNS = {"pos_reranker_raw_score", "neg_reranker_raw"}


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def compact_batch(job):
    """Write bounded Arrow batches; never accumulate the corpus in a table."""
    target = Path(job["output_dir"])
    if target.exists():
        raise ValueError("Compaction output already exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=target.name + ".partial-", dir=target.parent))
    output = staging / "mined.parquet"
    required = set(MINING_SCHEMA.names) - OPTIONAL_COLUMNS
    rows = 0
    # A worker is deliberately single-threaded and reads one input file at a time.
    pa.set_cpu_count(1)
    try:
        with pq.ParquetWriter(output, MINING_SCHEMA, compression="zstd", use_dictionary=False) as writer:
            for number, identity in enumerate(job["inputs"], 1):
                path = Path(identity["path"])
                if file_identity(path) != identity:
                    raise ValueError("Mining input changed before compaction: {}".format(path))
                with pq.ParquetFile(path) as source:
                    names = set(source.schema_arrow.names)
                    if not required <= names:
                        raise ValueError("Mining output missing columns: {} in {}".format(sorted(required - names), path))
                    columns = [name for name in MINING_SCHEMA.names if name in names]
                    for batch in source.iter_batches(batch_size=job["batch_size"], columns=columns, use_threads=False):
                        arrays = [batch.column(batch.schema.get_field_index(field.name)).cast(field.type)
                                  if field.name in names else pa.nulls(batch.num_rows, type=field.type)
                                  for field in MINING_SCHEMA]
                        writer.write_table(pa.Table.from_arrays(arrays, schema=MINING_SCHEMA),
                                           row_group_size=job["batch_size"])
                        rows += batch.num_rows
                if file_identity(path) != identity:
                    raise ValueError("Mining input changed during compaction: {}".format(path))
                if number == 1 or number == len(job["inputs"]) or number % 16 == 0:
                    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
                    LOG.info("Compaction worker %d: file %d/%d; %s rows; peak %.2f GiB",
                             os.getpid(), number, len(job["inputs"]), format(rows, ",d"), peak_bytes / (1024 ** 3))
        checksum = sha256(output)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest = {"version": CACHE_VERSION, "inputs": job["inputs"], "num_rows": rows,
                    "worker_pid": os.getpid(), "worker_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
                    "size": output.stat().st_size, "mtime_ns": output.stat().st_mtime_ns,
                    "sha256": checksum, "path": "mined.parquet"}
        write_json(staging / "_SUCCESS.json", manifest)
        os.rename(staging, target)
        LOG.info("Compacted %d input files into %s (%s rows, %.1f MiB)", len(job["inputs"]),
                 target / "mined.parquet", format(rows, ",d"), manifest["size"] / (1024 ** 2))
        return manifest
    except BaseException:
        LOG.exception("Compaction failed; incomplete files retained at %s", staging)
        raise


def completed_batch(directory, inputs):
    """Reuse atomically completed batches only while their inputs/files match.

    Matching size and mtime avoid rehashing the full cache on every retry. The
    checksum is recorded for auditing; it is not an integrity guarantee if an
    external edit deliberately preserves both file size and modification time.
    """
    marker = directory / "_SUCCESS.json"
    if not marker.is_file():
        return None
    manifest = json.loads(marker.read_text())
    output = directory / "mined.parquet"
    if (manifest.get("version") != CACHE_VERSION or manifest.get("inputs") != inputs
            or not output.is_file() or manifest.get("path") != "mined.parquet"):
        raise ValueError("Invalid completed mining cache batch: {}".format(directory))
    stat = output.stat()
    if stat.st_size != manifest["size"] or stat.st_mtime_ns != manifest["mtime_ns"]:
        raise ValueError("Completed mining cache file changed: {}".format(output))
    return manifest


def compact_mining(paths, cache_dir, files_per_batch=64, batch_size=1024, stage=None):
    """Return compacted Parquets; completed worker batches survive retries."""
    if not paths or files_per_batch <= 0 or batch_size <= 0:
        raise ValueError("Compaction requires input files and positive batch sizes")
    identities = [file_identity(path) for path in paths]
    identity = {"version": CACHE_VERSION, "files_per_batch": files_per_batch,
                "batch_size": batch_size, "inputs": identities}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir) / ("mining-" + fingerprint)
    root.mkdir(parents=True, exist_ok=True)
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=True)
    LOG.info("Persistent mining cache: %s", root)
    outputs, total = [], 0
    count = (len(paths) + files_per_batch - 1) // files_per_batch
    for start in range(0, len(paths), files_per_batch):
        number = start // files_per_batch
        inputs = identities[start:start + files_per_batch]
        directory = root / "batch_{:05d}".format(number)
        manifest = completed_batch(directory, inputs)
        reused = manifest is not None
        if not reused:
            job_path = jobs / "batch_{:05d}.json".format(number)
            job = {"inputs": inputs, "batch_size": batch_size, "output_dir": str(directory),
                   "log_level": "INFO" if LOG.isEnabledFor(logging.INFO) else "WARNING"}
            write_json(job_path, job)
            label = "Compact mining batch {}/{} (files {}-{} of {})".format(
                number + 1, count, start + 1, start + len(inputs), len(paths))
            with stage(label) if stage else nullcontext():
                subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                "--job-json", str(job_path)], check=True)
            manifest = completed_batch(directory, inputs)
        total += manifest["num_rows"]
        outputs.append(directory / "mined.parquet")
        LOG.info("%s mining batch %d/%d: %s rows; %s cumulative", "Reused" if reused else "Completed",
                 number + 1, count, format(manifest["num_rows"], ",d"), format(total, ",d"))
    LOG.info("Mining compaction ready: %s rows in %d files", format(total, ",d"), len(outputs))
    return outputs, total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args(argv)
    job = json.loads(args.job_json.read_text())
    logging.basicConfig(level=getattr(logging, job.get("log_level", "INFO")),
                        format="%(asctime)s %(levelname)s %(message)s")
    compact_batch(job)


if __name__ == "__main__":
    main()
