#!/usr/bin/env python3
"""Aggregate generation origins in isolated, restartable hash partitions."""

import argparse
from contextlib import nullcontext
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from compact_release_mining import file_identity
from release_data import write_json


LOG = logging.getLogger(__name__)
CACHE_VERSION = 2
SCHEMA = pa.schema([("finepdf_id", pa.string()), ("query", pa.string()),
                    ("generation_metadata", pa.string())])


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def source_sql(paths):
    # Bucket directories are internal routing, not additional input columns.
    return "read_parquet([{}], hive_partitioning=false)".format(
        ",".join(literal(path) for path in paths))


def peak_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def cached_result(directory, identity):
    marker = directory / "_SUCCESS.json"
    if not marker.is_file():
        return None
    result = json.loads(marker.read_text())
    if result.get("identity") != identity:
        raise ValueError("Generation cache identity changed: {}".format(directory))
    for record in result["files"]:
        if file_identity(directory / record["name"]) != record["identity"]:
            raise ValueError("Completed generation cache file changed: {}".format(record["name"]))
    return result


def run_worker(job):
    """A fresh process releases native sort and Parquet allocations on exit."""
    target = Path(job["output_dir"])
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=target.name + ".partial-", dir=target.parent))
    for record in job["inputs"]:
        if file_identity(record["path"]) != record:
            raise ValueError("Generation input changed: {}".format(record["path"]))
    paths = [record["path"] for record in job["inputs"]]
    pa.set_cpu_count(1)
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit={}".format(literal(job["memory_limit"])))
        connection.execute("SET threads=1")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET temp_directory={}".format(literal(staging / "spill")))
        LOG.info("Generation %s worker %d: DuckDB limit %s; one thread; %s input rows",
                 job["kind"], os.getpid(), job["memory_limit"], format(job["input_rows"], ",d"))
        if job["kind"] == "partition":
            # Keep every bucket's writer open until COPY completes. A smaller
            # open-file cap repeatedly evicts writers as hash buckets alternate,
            # producing a new tiny Parquet on each reopen and exhausting file
            # quotas. One thread and small flush/row-group sizes bound buffers;
            # the output contains at most one file per nonempty bucket.
            connection.execute("SET partitioned_write_max_open_files={}".format(job["partitions"]))
            connection.execute("SET partitioned_write_flush_threshold=16384")
            connection.execute(
                "COPY (SELECT finepdf_id,query,origin_json,sequence,hash(finepdf_id)%{} AS bucket "
                "FROM {}) TO {} (FORMAT PARQUET, PARTITION_BY(bucket), "
                "COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)".format(
                    job["partitions"], source_sql(paths), literal(staging / "data")))
            outputs = sorted((staging / "data").glob("bucket=*/*.parquet"))
            if len(outputs) != len({path.parent for path in outputs}):
                raise ValueError("Partition writer produced multiple files per bucket; refusing to cache fragmented output")
            rows = sum(pq.ParquetFile(path).metadata.num_rows for path in outputs)
            origins = rows
        else:
            # Equal document/query keys always share a bucket. Sequence keeps
            # every origin in its original order, even across input row groups.
            reader = connection.execute(
                "SELECT finepdf_id,query,origin_json,sequence FROM {} "
                "ORDER BY finepdf_id,query,sequence".format(source_sql(paths))
            ).fetch_record_batch(job["batch_size"])
            records = (row for batch in reader for row in batch.to_pylist())
            rows, origins, buffer, last_report = 0, 0, [], time.monotonic()
            output = staging / "generation.parquet"
            with pq.ParquetWriter(output, SCHEMA, compression="zstd", use_dictionary=False) as writer:
                for (identifier, query), group in itertools.groupby(
                        records, lambda row: (row["finepdf_id"], row["query"])):
                    metadata = [json.loads(row["origin_json"]) for row in group]
                    origins += len(metadata)
                    buffer.append({"finepdf_id": identifier, "query": query,
                                   "generation_metadata": json.dumps(metadata, ensure_ascii=False)})
                    rows += 1
                    if len(buffer) >= job["batch_size"]:
                        writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
                        buffer.clear()
                        if time.monotonic() - last_report >= 60:
                            LOG.info("Generation worker %d: %s pairs; peak %.2f GiB",
                                     os.getpid(), format(rows, ",d"), peak_bytes() / 1024 ** 3)
                            last_report = time.monotonic()
                if buffer:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
            outputs = [output]
        if origins != job["input_rows"]:
            raise ValueError("Generation origin count changed: expected {}, got {}".format(job["input_rows"], origins))
        for record in job["inputs"]:
            if file_identity(record["path"]) != record:
                raise ValueError("Generation input changed during aggregation: {}".format(record["path"]))
    finally:
        connection.close()
    # Record identities at their final paths, before atomically publishing.
    files = []
    for path in outputs:
        name = path.relative_to(staging).as_posix()
        identity = file_identity(path)
        identity["path"] = str((target / name).resolve())
        files.append({"name": name, "identity": identity,
                      "num_rows": pq.ParquetFile(path).metadata.num_rows})
    result = {"identity": job["identity"], "files": files, "num_rows": rows,
              "input_rows": origins, "worker_pid": os.getpid(), "worker_peak_rss_bytes": peak_bytes()}
    write_json(staging / "_SUCCESS.json", result)
    os.rename(staging, target)
    LOG.info("Generation %s worker %d finished: %s rows in %d files; peak %.2f GiB",
             job["kind"], os.getpid(), format(rows, ",d"), len(files), result["worker_peak_rss_bytes"] / 1024 ** 3)
    return result


def ensure_worker(root, name, job, stage=None):
    target = root / name
    job["output_dir"] = str(target)
    result = cached_result(target, job["identity"])
    if result is not None:
        LOG.info("Reusing completed generation %s: %s rows", name, format(result["num_rows"], ",d"))
        return result
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=True)
    job_path = jobs / (name + ".json")
    write_json(job_path, job)
    with stage("Generation " + name) if stage else nullcontext():
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                        "--job-json", str(job_path)], check=True)
    return cached_result(target, job["identity"])


def aggregate_origins(connection, input_path, output_path, batch_size, partitions=None,
                      memory_limit="8GB", cache_dir=None, stage=None):
    """Partition once, sort each bucket separately, then concatenate in batches.

    The connection argument is retained for callers of the original API; no
    full-corpus sort runs in that connection. Outputs need no global key order:
    the later metadata join explicitly orders by mining document/query IDs.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    input_identity = file_identity(input_path)
    with pq.ParquetFile(input_path) as source:
        input_rows = source.metadata.num_rows
    partitions = partitions if partitions is not None else min(128, max(1, math.ceil(input_rows / 2_000_000)))
    if partitions <= 0 or batch_size <= 0:
        raise ValueError("Generation partitions and batch size must be positive")
    identity = {"version": CACHE_VERSION, "input": input_identity, "partitions": partitions,
                "batch_size": batch_size, "duckdb_version": duckdb.__version__}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir) if cache_dir else output_path.parent.parent / "generation-cache"
    root = cache_dir / ("origins-" + fingerprint)
    root.mkdir(parents=True, exist_ok=True)
    LOG.info("Generation cache: %s; %s origins; %d partitions; worker limit %s",
             root, format(input_rows, ",d"), partitions, memory_limit)
    common = {"memory_limit": memory_limit, "batch_size": batch_size,
              "log_level": "INFO" if LOG.isEnabledFor(logging.INFO) else "WARNING"}
    buckets = {}
    if partitions == 1 or input_rows == 0:
        buckets[0] = ([input_identity], input_rows)
    else:
        result = ensure_worker(root, "partitioned", {
            **common, "kind": "partition", "inputs": [input_identity], "input_rows": input_rows,
            "partitions": partitions, "identity": identity}, stage)
        for record in result["files"]:
            bucket = int(Path(record["name"]).parent.name.split("=", 1)[1])
            inputs, count = buckets.setdefault(bucket, ([], 0))
            inputs.append(record["identity"])
            buckets[bucket] = inputs, count + record["num_rows"]
    outputs, total, origins = [], 0, 0
    for number, (bucket, (inputs, count)) in enumerate(sorted(buckets.items()), 1):
        LOG.info("Aggregate generation partition %d/%d: %s origins", number, len(buckets), format(count, ",d"))
        result = ensure_worker(root, "aggregated_{:05d}".format(bucket), {
            **common, "kind": "aggregate", "inputs": inputs, "input_rows": count,
            "identity": {"parent": identity, "bucket": bucket}}, stage)
        outputs.extend(root / "aggregated_{:05d}".format(bucket) / record["name"] for record in result["files"])
        total += result["num_rows"]
        origins += result["input_rows"]
    if origins != input_rows or file_identity(input_path) != input_identity:
        raise ValueError("Generation inputs changed during partitioned aggregation")
    # Publish the single compatibility file only after every partition succeeds.
    # This is a streaming concatenation, with no second global sort or merge.
    partial = output_path.with_name(output_path.name + ".partial-" + fingerprint[:12])
    written = 0
    with pq.ParquetWriter(partial, SCHEMA, compression="zstd", use_dictionary=False) as writer:
        for number, path in enumerate(outputs, 1):
            with pq.ParquetFile(path) as source:
                for batch in source.iter_batches(batch_size=max(batch_size, 8192), use_threads=False):
                    writer.write_batch(batch)
                    written += batch.num_rows
            LOG.info("Combined generation partition %d/%d: %s cumulative pairs", number, len(outputs), format(written, ",d"))
    if written != total:
        raise ValueError("Combined generation row count differs from completed partitions")
    os.replace(partial, output_path)
    LOG.info("Aggregated %s distinct document/query pairs from %s origins to %s",
             format(total, ",d"), format(input_rows, ",d"), output_path)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job_json.read_text())
    logging.basicConfig(level=getattr(logging, job.get("log_level", "INFO")),
                        format="%(asctime)s %(levelname)s %(message)s")
    run_worker(job)


if __name__ == "__main__":
    main()
