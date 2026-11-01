#!/usr/bin/env python3
"""Export grouped hard negatives as a references-only, pinned FinePDFs release.

Integer references address the exact ordered source manifest, not mining shards.
DuckDB spills large joins/sorts to disk; Arrow streams the remaining work.
Nothing is uploaded, and source artifacts are never modified.
"""

import argparse
import base64
from contextlib import contextmanager
import glob
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from release_data import RELEASE_FORMAT_VERSION, REF_SCHEMA, TRAIN_SCHEMA, NEGATIVE_ALIAS_SCHEMA, ShardedWriter, sha256, write_json
from compact_release_mining import compact_mining, file_identity
from aggregate_release_generation import aggregate_origins
from release_training_workers import COUNTERS, training_batches
from resolve_finepdf_source import resolve_ambiguous_ids
from generation_validation import open_generation, generation_batches, validate_generation
from metadata_validation import missing_metadata
from generation_query_encoding import query_text, recover_query_keys, serialized_query_literal


LOG = logging.getLogger(__name__)
DEFAULT_METADATA_FIELDS = (
    "stage1_question_type,stage1_difficulty,stage1_character,query_length_target,"
    "stage1_raw,stage2_raw,request_id,time_taken"
)


def memory_status():
    """Linux process memory, including allocations outside DuckDB's limit."""
    try:
        fields = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _ = line.split()
                fields[key.rstrip(":")] = int(value) / (1024 ** 2)
        return "RSS {:.2f} GiB, peak {:.2f} GiB".format(fields["VmRSS"], fields["VmHWM"])
    except (OSError, KeyError, ValueError):
        return "RSS unavailable"


@contextmanager
def log_stage(label):
    """Report elapsed time even while a DuckDB operation blocks Python progress."""
    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(60):
            LOG.info("%s: still running (%.1f min elapsed; %s)",
                     label, (time.monotonic() - started) / 60, memory_status())

    LOG.info("Starting: %s (%s)", label, memory_status())
    worker = threading.Thread(target=heartbeat, name="release-progress", daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        LOG.error("Failed: %s (%.1f s; %s)", label, time.monotonic() - started, memory_status())
        raise
    else:
        LOG.info("Finished: %s (%.1f s; %s)", label, time.monotonic() - started, memory_status())
    finally:
        stopped.set()
        worker.join()


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def parquet_sql(paths):
    return "read_parquet([{}], union_by_name=true)".format(
        ",".join(sql_literal(str(p)) for p in paths))


def matching_files(pattern):
    files = sorted(Path(p).resolve() for p in glob.glob(str(pattern), recursive=True)
                   if Path(p).is_file())
    if not files:
        raise ValueError("No files match {}".format(pattern))
    return files


def generation_inputs(args):
    paths = matching_files(args.queries_glob)
    excluded = {path.resolve() for path in args.exclude_generation_file}
    unknown = excluded - set(paths)
    if unknown:
        raise ValueError("Excluded generation files do not match --queries-glob: {}".format(
            ", ".join(str(path) for path in sorted(unknown))))
    records = []
    for path in sorted(excluded):
        stat = path.stat()
        records.append({"path": str(path), "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns, "reason": "explicit_exclusion"})
        LOG.warning("Explicitly excluding generation input: %s; mined query metadata coverage is still required", path)
    paths = [path for path in paths if path not in excluded]
    if not paths:
        raise ValueError("No generation files remain after explicit exclusions")
    with log_stage("Preflight generation Parquet footers and schemas"):
        report = validate_generation(paths)
    return paths, {"file_count": len(paths), "input_rows": sum(r["num_rows"] for r in report["files"]),
                   "all_generation_inputs_included": not records, "excluded_files": records}


def load_json(path):
    return json.loads(Path(path).read_text())


def discover_mining(root, embeddings, allow_partial=False, query_counts=None):
    """Prefer materialized outputs; require completion evidence for every shard."""
    root = Path(root)
    if not list(root.glob("shard_*")) and (root / "v5_http").is_dir():
        root = root / "v5_http"
    complete = set()
    for directory in {root, root.parent if root.name == "v5_http" else root}:
        for checkpoint in directory.glob("checkpoint*rank*.json"):
            complete.update(int(i) for i in load_json(checkpoint).get("pass2_completed", []))
    files, missing = [], []
    for position, embedding in enumerate(embeddings):
        directory = root / "shard_{:05d}".format(position)
        if (pq.ParquetFile(embedding).metadata.num_rows == 0
                or (query_counts is not None and query_counts[position] == 0)):
            continue
        materialized = directory / "hard_negatives.parquet"
        if (directory / "_MATERIALIZED_SUCCESS").is_file() and materialized.is_file():
            files.append(materialized)
            LOG.info("Mining shard %d/%d: completed materialized file %s",
                     position + 1, len(embeddings), materialized)
            continue
        chunk_meta = directory / "_chunked_meta.json"
        if chunk_meta.is_file():
            count = int(load_json(chunk_meta)["num_chunks"])
            parts = []
            valid = count >= 0
            for chunk in range(count):
                chunk_dir = directory / "chunk_{:05d}".format(chunk)
                chunk_files = sorted(chunk_dir.glob("*.parquet"))
                valid = valid and (chunk_dir / "_SUCCESS").is_file() and bool(chunk_files)
                parts.extend(chunk_files)
            if valid:
                files.extend(parts)
                LOG.info("Mining shard %d/%d: %d completed chunks, %d files",
                         position + 1, len(embeddings), count, len(parts))
                continue
        elif position in complete:
            parts = sorted(directory.glob("*.parquet"))
            if parts:
                files.extend(parts)
                LOG.info("Mining shard %d/%d: %d completed legacy files",
                         position + 1, len(embeddings), len(parts))
                continue
        missing.append(position)
    if missing and not allow_partial:
        raise ValueError("Incomplete mining shards: {}. Finish mining or explicitly use "
                         "--allow-partial for a partial release.".format(missing))
    if not files:
        raise ValueError("No completed mining output found in {}".format(root))
    LOG.info("Mining discovery: %d input files, %d missing shards", len(files), len(missing))
    return files, missing


def source_manifest(files, root):
    entries, offset = [], 0
    for position, path in enumerate(files, 1):
        relative = path.relative_to(root).as_posix()
        pf = pq.ParquetFile(path)
        if "id" not in pf.schema_arrow.names or "text" not in pf.schema_arrow.names:
            raise ValueError("FinePDFs shard lacks id/text: {}".format(path))
        count = pf.metadata.num_rows
        entries.append({"path": relative, "start_row": offset, "num_rows": count})
        offset += count
        LOG.info("Source shard %d/%d: %s (%s rows; %s cumulative)",
                 position, len(files), relative, format(count, ",d"), format(offset, ",d"))
    return entries, offset


def stage_generation(paths, output, fields, batch_size, stats=None, excluded_query_keys=None):
    """Handle raw scalar queries and the compactor's aligned per-query lists.

    Read only identifiers, query text, and explicitly selected metadata fields.
    Compaction loses original query suffixes: do not invent source query IDs.
    """
    schema = pa.schema([("finepdf_id", pa.string()), ("query", pa.string()),
                        ("origin_json", pa.string()), ("sequence", pa.int64())])
    sequence = 0
    binary_queries = 0
    pickled_queries = 0
    excluded_queries = 0
    excluded_query_keys = excluded_query_keys or set()
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        for file_number, path in enumerate(paths, 1):
            LOG.info("Opening generation file %d/%d: %s", file_number, len(paths), path)
            pf = open_generation(path)
            file_start, processed = sequence, 0
            last_report = time.monotonic()
            LOG.info("Generation file %d/%d: %s (%s input rows)",
                     file_number, len(paths), path, format(pf.metadata.num_rows, ",d"))
            names = set(pf.schema_arrow.names)
            if not {"id", "query"} <= names:
                raise ValueError("Generation input must contain id and query: {}".format(path))
            columns = ["id", "query"] + [f for f in fields if f in names and f not in {"id", "query"}]
            for batch in generation_batches(path, pf, columns, batch_size):
                rows = []
                for row_number, record in enumerate(batch.to_pylist(), start=processed):
                    raw_id = record["id"]
                    if not isinstance(raw_id, str) or not raw_id:
                        raise ValueError("Missing generation document ID in {}".format(path))
                    finepdf_id = raw_id.rsplit("__s1d_", 1)[0]
                    queries = record["query"]
                    compacted = isinstance(queries, list)
                    queries = queries if compacted else [queries]
                    for field in columns[2:]:
                        value = record[field]
                        if isinstance(value, list) and (not compacted or len(value) != len(queries)):
                            raise ValueError("Unaligned metadata {} in {}".format(field, path))
                    for position, query in enumerate(queries):
                        if query is None:
                            continue
                        key = repr(query) if isinstance(query, bytes) else str(query).strip()
                        try:
                            text = query_text(query)
                        except UnicodeError as exc:
                            if (finepdf_id, key) in excluded_query_keys:
                                excluded_queries += 1
                                continue
                            raise ValueError("Unsupported binary generation query in {} at row {}, query position {}: "
                                             "expected UTF-8 or a primitive serialized string".format(
                                                 path, row_number, position)) from exc
                        if not text:
                            continue
                        pickled = isinstance(query, bytes) and query.startswith(b"\x80")
                        binary_queries += isinstance(query, bytes) and not pickled
                        pickled_queries += pickled
                        metadata = {field: (record[field][position]
                                    if isinstance(record[field], list) else record[field])
                                    for field in columns[2:]}
                        metadata["source_query_id"] = None if compacted else raw_id
                        if pickled:
                            metadata["query_encoding_recovery"] = {
                                "format": "pickle_string", "original_query_base64": base64.b64encode(query).decode("ascii")}
                        rows.append({"finepdf_id": finepdf_id, "query": text,
                                     "origin_json": json.dumps(metadata, ensure_ascii=False,
                                                               sort_keys=True, allow_nan=False),
                                     "sequence": sequence})
                        sequence += 1
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                processed += batch.num_rows
                now = time.monotonic()
                if now - last_report >= 60:
                    LOG.info("Generation file %d/%d: %s/%s input rows; %s total query origins",
                             file_number, len(paths), format(processed, ",d"),
                             format(pf.metadata.num_rows, ",d"), format(sequence, ",d"))
                    last_report = now
            pf.close()
            LOG.info("Generation file %d/%d finished: %s query origins; %s cumulative",
                     file_number, len(paths), format(sequence - file_start, ",d"), format(sequence, ",d"))
    LOG.info("Staged %s generation query origins to %s", format(sequence, ",d"), output)
    LOG.info("Decoded %s binary generation queries as UTF-8", format(binary_queries, ",d"))
    LOG.info("Decoded %s binary generation queries as primitive serialized strings", format(pickled_queries, ",d"))
    LOG.info("Omitted %s invalid binary generation origins matching excluded serialized mining queries",
             format(excluded_queries, ",d"))
    if stats is not None:
        stats["binary_queries_decoded_as_utf8"] = binary_queries
        stats["binary_queries_decoded_as_pickle_strings"] = pickled_queries
        stats["excluded_serialized_query_origins"] = excluded_queries
    return sequence


def require_zero(connection, sql, message):
    with log_stage("Validate: " + message):
        count = connection.execute(sql).fetchone()[0]
        if count:
            raise ValueError("{} ({} rows)".format(message, count))


def load_mining(connection, paths, files_per_batch, cache_dir=None, batch_size=1024):
    """Query compacted Parquets without appending nested arrays into DuckDB."""
    if cache_dir is None:
        database = connection.execute("PRAGMA database_list").fetchone()[2]
        if not database:
            raise ValueError("Set cache_dir when using an in-memory DuckDB connection")
        cache_dir = Path(database).parent / "mining-cache"
    compacted, total_rows = compact_mining(paths, cache_dir, files_per_batch, batch_size, stage=log_stage)
    # All worker outputs have the same schema. Only compacted file metadata is
    # bound here, and no full-corpus nested INSERT table is ever constructed.
    source = "read_parquet([{}], union_by_name=false)".format(
        ",".join(sql_literal(path) for path in compacted))
    LOG.info("Opening %d compacted mining files (%s rows; %s)",
             len(compacted), format(total_rows, ",d"), memory_status())
    connection.execute("CREATE VIEW mined_all AS SELECT * FROM {}".format(source))
    connection.execute("CREATE VIEW mined AS SELECT * FROM mined_all")
    return total_rows


def filter_serialized_query_groups(connection, args):
    """Exclude complete groups whose historical teacher input was serialized bytes."""
    # Older saved runs have no raw view. Copy its SELECT, rather than making a
    # view that would become circular when the temporary filtered view replaces it.
    if not connection.execute("SELECT count(*) FROM duckdb_views() WHERE view_name='mined_all'").fetchone()[0]:
        sql = connection.execute("SELECT sql FROM duckdb_views() WHERE view_name='mined'").fetchone()[0]
        connection.execute("CREATE TEMP VIEW mined_all AS " + sql.split(" AS ", 1)[1])
    prefixes = [sql_literal("b'\\x80"), sql_literal('b"\\x80')]
    reader = connection.execute(
        "SELECT doc_id_int,query FROM (SELECT doc_id_int,unnest(query_texts) AS query FROM mined_all) q "
        "WHERE starts_with(query,{}) OR starts_with(query,{})".format(*prefixes)
    ).fetch_record_batch(args.batch_size)
    affected = {row["doc_id_int"] for batch in reader for row in batch.to_pylist()
                if serialized_query_literal(row["query"])}
    args.query_group_exclusion = {"policy": args.serialized_query_policy, "excluded_groups": 0, "excluded_queries": 0}
    args.excluded_generation_query_keys = set()
    if not affected:
        return
    if args.serialized_query_policy != "exclude-groups":
        raise ValueError("{} mining query groups contain serialized byte-string queries. "
                         "Their teacher scores were computed for those strings. Use "
                         "--serialized-query-policy exclude-groups to exclude the affected groups.".format(len(affected)))
    connection.register("excluded_query_group_ids", pa.table({"doc_id_int": sorted(affected)}))
    connection.execute("CREATE TEMP VIEW mined AS SELECT m.* FROM mined_all m "
                       "ANTI JOIN excluded_query_group_ids e USING(doc_id_int)")
    mapping = parquet_sql([args.index_dir / "doc_id_map.parquet"])
    records = connection.execute(
        "SELECT m.doc_id_int,i.doc_id_str AS finepdf_id,m.qids,m.query_texts FROM mined_all m "
        "JOIN excluded_query_group_ids e USING(doc_id_int) JOIN {} i USING(doc_id_int) "
        "ORDER BY m.doc_id_int".format(mapping)).fetchall()
    if len(records) != len(affected):
        raise ValueError("Cannot identify every excluded mining query group")
    report_records = []
    for doc_id, identifier, qids, queries in records:
        for query in queries:
            args.excluded_generation_query_keys.add((identifier, query))
        report_records.append({"doc_id_int": doc_id, "finepdf_id": identifier,
                               "mining_query_ids": qids, "query_count": len(queries)})
    args.query_group_exclusion.update(excluded_groups=len(records), excluded_queries=sum(len(r[3]) for r in records))
    write_json(args.work / "excluded_query_groups.json", {**args.query_group_exclusion, "groups": report_records})
    LOG.warning("Excluded %s query groups (%s queries) whose historical mining inputs were serialized bytes; "
                "retained teacher scores and mining inputs are unchanged",
                format(len(records), ",d"), format(args.query_group_exclusion["excluded_queries"], ",d"))


def excluded_empty_documents(connection, args):
    """Allow only queryless, textless counter IDs to be omitted from candidates."""
    missing = connection.execute(
        "SELECT m.* FROM mining_ids m ANTI JOIN doc_mapping d USING(old_idx) ORDER BY old_idx"
    ).fetchall()
    if not missing:
        return set()
    if args.missing_document_policy == "error":
        raise ValueError("Mining document missing from FinePDFs ({} rows); examples: {}. "
                         "Use --missing-document-policy drop-empty-negatives only for "
                         "queryless, empty counter-ID artifacts.".format(len(missing), missing[:5]))
    missing_ids = {i: identifier for i, identifier in missing}
    require_zero(connection, "SELECT count(*) FROM mined m ANTI JOIN doc_mapping d "
                 "ON m.doc_id_int=d.old_idx", "Missing FinePDFs document is a training positive")
    if any(not re.fullmatch(r"[0-9]+", identifier or "") for identifier in missing_ids.values()):
        raise ValueError("Missing documents include non-counter IDs; restore the correct FinePDFs source")
    offset, verified = 0, set()
    directory = args.embeddings_dir or args.index_dir / "embeddings"
    for path in sorted(directory.glob("shard_*.parquet")):
        with pq.ParquetFile(path) as pf:
            selected = {i - offset: identifier for i, identifier in missing_ids.items()
                        if offset <= i < offset + pf.metadata.num_rows}
            if selected:
                required = {"doc_id_str", "pos_text", "all_query_texts", "repr_query_text"}
                if not required <= set(pf.schema_arrow.names):
                    raise ValueError("Cannot verify empty candidate artifacts in {}".format(path))
                base = 0
                for group in range(pf.metadata.num_row_groups):
                    end = base + pf.metadata.row_group(group).num_rows
                    wanted = {row for row in selected if base <= row < end}
                    if wanted:
                        LOG.info("Checking %d missing candidate documents in %s row group %d",
                                 len(wanted), path, group)
                        position = base
                        for batch in pf.iter_batches(batch_size=512, row_groups=[group],
                                                     columns=sorted(required), use_threads=False):
                            for row in sorted(wanted):
                                if position <= row < position + batch.num_rows:
                                    record = batch.slice(row - position, 1).to_pylist()[0]
                                    if (record["doc_id_str"] != selected[row] or record["pos_text"] != ""
                                            or record["all_query_texts"] != [] or record["repr_query_text"] != ""):
                                        raise ValueError("Missing document {} is not a verified empty, queryless "
                                                         "counter-ID artifact".format(offset + row))
                                    verified.add(offset + row)
                                    wanted.remove(row)
                            position += batch.num_rows
                            if not wanted:
                                break
                    base = end
            offset += pf.metadata.num_rows
    if verified != set(missing_ids):
        raise ValueError("Missing document IDs fall outside the canonical embedding corpus")
    report = args.work / "source-identity-report"
    report.mkdir(exist_ok=True)
    write_json(report / "excluded_empty_documents.json", {
        "policy": args.missing_document_policy,
        "documents": [{"old_idx": i, "counter_id": missing_ids[i]} for i in sorted(verified)],
        "verification": "Exact empty canonical text, no canonical queries, and no mined positive."})
    LOG.warning("Excluding %d verified empty counter-ID documents from candidate lists; "
                "all training positives and queries are retained", len(verified))
    return verified


def build_mapping(connection, args, mining_files, source_files, entries):
    load_mining(connection, mining_files, args.mining_files_per_batch,
                args.mining_cache_dir or args.work.parent / "mining-cache", args.batch_size)
    LOG.info("Validating mined rows and collecting referenced document IDs")
    require_zero(connection, "SELECT count(*) - count(DISTINCT doc_id_int) FROM mined",
                 "Duplicate positive mining IDs (overlapping output artifacts)")
    require_zero(connection, "SELECT count(*) FROM mined WHERE query_texts IS NULL OR qids IS NULL "
                 "OR len(query_texts)=0 OR len(query_texts) != len(qids)", "Invalid grouped queries")
    filter_serialized_query_groups(connection, args)
    with log_stage("Collect referenced mining document IDs"):
        connection.execute("CREATE TABLE needed AS SELECT doc_id_int AS old_idx FROM mined UNION "
                           "SELECT unnest(list_transform(neg_embed, x -> x.doc_id)) FROM mined UNION "
                           "SELECT unnest(list_transform(neg_reranker, x -> x.doc_id)) FROM mined UNION "
                           "SELECT unnest(list_transform(skipped_doc_ids, x -> x.doc_id)) FROM mined")
    require_zero(connection, "SELECT count(*) FROM needed WHERE old_idx IS NULL OR old_idx < 0",
                 "Missing or negative mining document ID")
    mapping_file = args.index_dir / "doc_id_map.parquet"
    LOG.info("Joining referenced IDs to %s", mapping_file)
    connection.execute("CREATE TABLE mining_ids AS SELECT m.doc_id_int AS old_idx, "
                       "m.doc_id_str AS finepdf_id FROM {} m JOIN needed n ON n.old_idx=m.doc_id_int"
                       .format(parquet_sql([mapping_file])))
    require_zero(connection, "SELECT count(*) FROM needed n LEFT JOIN mining_ids m USING(old_idx) "
                 "WHERE m.old_idx IS NULL", "Mining ID missing from doc_id_map")
    require_zero(connection, "SELECT count(*) - count(DISTINCT old_idx) FROM mining_ids",
                 "Ambiguous mining ID map")
    LOG.info("Scanning IDs in %d source files and joining referenced mining documents", len(source_files))
    if args.source_index_dir:
        connection.execute("CREATE TABLE source_matches AS SELECT s.* FROM {} s "
                           "SEMI JOIN mining_ids m USING(finepdf_id)".format(parquet_sql(source_files)))
    else:
        manifest_table = pa.Table.from_pylist([
            {"filename": str(path), "source_shard": entry["path"], "start_row": entry["start_row"]}
            for path, entry in zip(source_files, entries)])
        connection.register("source_files", manifest_table)
        source_sql = "read_parquet([{}], filename=true, file_row_number=true)".format(
            ",".join(sql_literal(p) for p in source_files))
        connection.execute("CREATE TABLE source_matches AS SELECT s.id AS finepdf_id, "
                           "f.start_row+s.file_row_number AS finepdf_idx, f.source_shard, "
                           "s.file_row_number AS row_in_shard FROM {} s "
                           "JOIN source_files f ON s.filename=f.filename "
                           "SEMI JOIN mining_ids m ON s.id=m.finepdf_id".format(source_sql))
    LOG.info("Source scan finished; checking source coordinates and resolving repeated IDs")
    require_zero(connection, "SELECT count(*) - count(DISTINCT finepdf_idx) FROM source_matches",
                 "Source index maps different documents to the same position")
    connection.register("source_coordinates", pa.Table.from_pylist([
        {"source_shard": entry["path"], "start_row": entry["start_row"], "num_rows": entry["num_rows"]}
        for entry in entries]))
    require_zero(connection, "SELECT count(*) FROM source_matches s LEFT JOIN source_coordinates c "
                 "USING(source_shard) WHERE c.source_shard IS NULL OR s.row_in_shard IS NULL "
                 "OR s.finepdf_idx IS NULL OR s.row_in_shard<0 OR s.row_in_shard>=c.num_rows "
                 "OR s.finepdf_idx != c.start_row+s.row_in_shard", "Invalid source-index coordinates")
    with log_stage("Resolve repeated source IDs using exact canonical text"):
        resolved, args.source_identity_resolution = resolve_ambiguous_ids(
            connection, args.finepdf_root, args.index_dir,
            args.embeddings_dir or args.index_dir / "embeddings", args.work / "source-identity-report")
    connection.execute("CREATE TABLE doc_mapping AS SELECT m.old_idx,s.* FROM mining_ids m "
                       "JOIN source_matches s USING(finepdf_id) "
                       "ANTI JOIN ambiguous_source_ids a USING(finepdf_id)")
    connection.register("resolved_source_duplicates", resolved)
    connection.execute("INSERT INTO doc_mapping (old_idx,finepdf_id,finepdf_idx,source_shard,row_in_shard) "
                       "SELECT old_idx,finepdf_id,finepdf_idx,source_shard,row_in_shard "
                       "FROM resolved_source_duplicates")
    connection.unregister("resolved_source_duplicates")
    args.excluded_document_ids = excluded_empty_documents(connection, args)
    require_zero(connection, "SELECT count(*)-count(DISTINCT old_idx) FROM doc_mapping",
                 "Mining document maps to multiple source rows")
    args.source_identity_resolution["aliased_mining_documents"] = connection.execute(
        "SELECT count(*)-count(DISTINCT finepdf_idx) FROM doc_mapping").fetchone()[0]
    # UUIDs can repeat, but public row positions remain distinct. Keep every
    # selected variant, and omit unselected aliases from document_refs only.
    connection.execute("CREATE OR REPLACE TABLE source_matches AS "
                       "SELECT DISTINCT finepdf_id,finepdf_idx,source_shard,row_in_shard FROM doc_mapping")
    max_id = connection.execute("SELECT max(old_idx) FROM needed").fetchone()[0]
    if max_id is None or max_id < 0:
        raise ValueError("No valid referenced document IDs")
    if max_id >= args.mining_doc_count:
        raise ValueError("Mining ID exceeds the canonical embedding corpus size")
    # A disk-backed integer array makes per-example negative remapping O(1).
    # It contains no Python corpus-sized dictionary and is not a public artifact.
    import numpy as np
    map_path = args.work / "old_to_source.int64"
    LOG.info("Writing ID remapping: %s (%s slots; %.2f GiB)", map_path,
             format(max_id + 1, ",d"), (max_id + 1) * 8 / (1024 ** 3))
    remap = np.memmap(map_path, mode="w+", dtype="int64", shape=(max_id + 1,))
    remap[:] = -1
    reader = connection.execute("SELECT old_idx,finepdf_idx FROM doc_mapping").fetch_record_batch(args.batch_size)
    mapped_count = 0
    for batch in reader:
        old = batch.column(0).to_numpy()
        if (old < 0).any():
            raise ValueError("Negative mining document ID")
        remap[old] = batch.column(1).to_numpy()
        mapped_count += batch.num_rows
    remap.flush()
    LOG.info("Mapped %s referenced mining documents to source positions", format(mapped_count, ",d"))
    return remap


def resolve_query_metadata(connection, args, generation_path):
    LOG.info("Validating query IDs and joining generation metadata")
    generation_paths = [generation_path] if isinstance(generation_path, (str, Path)) else generation_path
    connection.execute("CREATE OR REPLACE TEMP VIEW generation AS SELECT * FROM {}".format(parquet_sql(generation_paths)))
    connection.execute("CREATE OR REPLACE TEMP VIEW mining_queries AS SELECT doc_id_int, unnest(query_texts) AS query, "
                       "unnest(qids) AS mining_query_id, generate_subscripts(query_texts,1) AS position FROM mined")
    require_zero(connection, "SELECT count(*) FROM mining_queries WHERE query IS NULL OR trim(query)='' "
                 "OR mining_query_id IS NULL", "Empty query or missing mining query ID")
    require_zero(connection, "SELECT count(*) - count(DISTINCT mining_query_id) FROM mining_queries",
                 "Duplicate mining query IDs")
    with log_stage("Validate: Generation metadata coverage"):
        report_dir = args.work / "metadata-coverage-report"
        coverage = missing_metadata(connection, generation_path, report_dir)
        if coverage["missing_queries"]:
            raise ValueError("Missing generation metadata for mined queries ({} rows). "
                             "Inspect {} before choosing recovery inputs.".format(coverage["missing_queries"], report_dir))
    path = args.work / "resolved_queries.parquet"
    connection.execute("COPY (SELECT q.*, d.finepdf_idx AS positive_idx, g.generation_metadata "
                       "FROM mining_queries q JOIN doc_mapping d ON q.doc_id_int=d.old_idx "
                       "JOIN generation g ON d.finepdf_id=g.finepdf_id AND q.query=g.query "
                       "ORDER BY d.finepdf_idx, q.doc_id_int, q.position) TO {} "
                       "(FORMAT PARQUET, COMPRESSION ZSTD)".format(sql_literal(path)))
    LOG.info("Wrote resolved query metadata: %s", path)
    return path


def numeric_score(value):
    if value is None or not math.isfinite(float(value)):
        raise ValueError("Missing or non-finite score")
    return float(value)


def score_map(entries):
    if entries is None:
        raise ValueError("Missing candidate score list")
    result = {}
    for entry in entries:
        if entry["doc_id"] is None:
            raise ValueError("Missing candidate document ID")
        doc_id = int(entry["doc_id"])
        if doc_id in result:
            raise ValueError("Duplicate candidate document ID {}".format(doc_id))
        result[doc_id] = numeric_score(entry["score"])
    return result


def generation_origins(serialized):
    """Expose common fields natively in Arrow, retaining all selected metadata."""
    origins = []
    for metadata in json.loads(serialized):
        origins.append({
            "source_query_id": metadata.get("source_query_id"),
            "query_type": metadata.get("stage1_question_type"),
            "difficulty": metadata.get("stage1_difficulty"),
            "character": metadata.get("stage1_character"),
            "query_length_target": metadata.get("query_length_target"),
            "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        })
    return origins


def training_row(record, queries, remap, next_query_id, alias_policy="drop", removed_candidates=None,
                 excluded_document_ids=()):
    if record["doc_id_int"] in excluded_document_ids:
        raise ValueError("Cannot exclude a training positive")
    positive_idx = int(remap[record["doc_id_int"]])
    if [q["query"] for q in queries] != record["query_texts"]:
        raise ValueError("Query ordering changed while joining metadata")
    if [q["mining_query_id"] for q in queries] != record["qids"]:
        raise ValueError("Query ID ordering changed while joining metadata")
    embed = score_map(record["neg_embed"])
    rerank = score_map(record["neg_reranker"])
    if embed.keys() != rerank.keys():
        raise ValueError("Embedding/reranker candidate IDs disagree")
    raw = record.get("neg_reranker_raw")
    raw = score_map(raw) if raw is not None else None
    if raw is not None and raw.keys() != embed.keys():
        raise ValueError("Raw reranker candidate IDs disagree")
    if record["doc_id_int"] in embed:
        raise ValueError("Scored candidates contain the positive's own mining ID")
    skipped = score_map(record["skipped_doc_ids"])
    if record["doc_id_int"] in skipped:
        raise ValueError("Skipped neighbors contain the positive's own mining ID")
    if embed.keys() & skipped.keys():
        raise ValueError("Skipped and reranked candidates overlap")

    for scores, reason in ((embed, "empty_missing_negative"), (skipped, "empty_missing_skipped")):
        for doc_id in list(scores):
            if doc_id in excluded_document_ids:
                if removed_candidates is not None:
                    removed_candidates.append({
                        "query_id": next_query_id, "mining_query_id": queries[0]["mining_query_id"],
                        "positive_mining_id": record["doc_id_int"], "positive_idx": positive_idx,
                        "removed_mining_id": doc_id, "canonical_idx": None,
                        "retained_mining_id": None, "reason": reason,
                        "retrieval_score": scores[doc_id],
                        "reranker_score": rerank[doc_id] if reason == "empty_missing_negative" else None,
                        "reranker_raw_score": raw[doc_id] if raw is not None and reason == "empty_missing_negative" else None})
                del scores[doc_id]

    def mapped(ids):
        result = []
        for doc_id in ids:
            if doc_id < 0 or doc_id >= len(remap) or remap[doc_id] < 0:
                raise ValueError("Unresolved candidate ID {}".format(doc_id))
            result.append(int(remap[doc_id]))
        return result

    expanded = [{"query_id": next_query_id + i, "mining_query_id": q["mining_query_id"],
                 "query": q["query"], "query_word_count": len(q["query"].split()),
                 "generation_metadata": generation_origins(q["generation_metadata"])}
                for i, q in enumerate(queries)]
    # Ray aggregation need not preserve pair order. Restore a deterministic
    # hardness order instead of assuming the input lists remain kNN-sorted.
    canonical = dict(zip(embed, mapped(embed)))
    # Skipped slots are excluded mining neighbors, not training negatives.
    # Distinct mining IDs can alias a positive/scored document after exact
    # remapping. Preserve those audit slots independently of training negatives.
    mapped(skipped)
    negative_ids = sorted(embed, key=lambda i: (-embed[i], canonical[i], i))
    has_aliases = positive_idx in canonical.values() or len(set(canonical.values())) != len(canonical)
    if has_aliases and alias_policy == "error":
        raise ValueError("Source-ID remapping exposes self/duplicate candidates for positive mining ID {} "
                         "(source index {}); use --negative-alias-policy drop to remove canonical aliases".format(
                             record["doc_id_int"], positive_idx))
    if alias_policy not in {"drop", "error"}:
        raise ValueError("Unknown negative alias policy: {}".format(alias_policy))
    if has_aliases:
        retained, selected = [], {}
        for doc_id in negative_ids:
            index = canonical[doc_id]
            if index == positive_idx:
                reason, retained_id = "alias_of_positive", record["doc_id_int"]
            elif index in selected:
                reason, retained_id = "duplicate_negative_alias", selected[index]
            else:
                selected[index] = doc_id
                retained.append(doc_id)
                continue
            if removed_candidates is not None:
                removed_candidates.append({
                    "query_id": next_query_id, "mining_query_id": queries[0]["mining_query_id"],
                    "positive_mining_id": record["doc_id_int"], "positive_idx": positive_idx,
                    "removed_mining_id": doc_id, "canonical_idx": index,
                    "retained_mining_id": retained_id, "reason": reason,
                    "retrieval_score": embed[doc_id], "reranker_score": rerank[doc_id],
                    "reranker_raw_score": raw[doc_id] if raw is not None else None})
        negative_ids = retained
    skipped_ids = sorted(skipped, key=lambda i: (-skipped[i], int(remap[i])))
    return {
        **expanded[0], "other_queries": expanded[1:], "positive_idx": positive_idx,
        "negative_idxs": [canonical[i] for i in negative_ids], "negative_retrieval_scores": [embed[i] for i in negative_ids],
        "negative_reranker_scores": [rerank[i] for i in negative_ids],
        "negative_reranker_raw_scores": [raw[i] for i in negative_ids] if raw is not None else None,
        "positive_reranker_score": numeric_score(record["pos_reranker_score"]),
        "positive_reranker_raw_score": numeric_score(record["pos_reranker_raw_score"])
        if record.get("pos_reranker_raw_score") is not None else None,
        "skipped_idxs": mapped(skipped_ids), "skipped_retrieval_scores": [skipped[i] for i in skipped_ids],
    }


def export_train(connection, args, remap, query_path, root):
    LOG.info("Sorting mined examples by source position for training output")
    writer = ShardedWriter(root, "train", TRAIN_SCHEMA, args.rows_per_shard)
    workers = args.train_workers
    LOG.info("Training conversion: %d CPU worker(s), at most %d batches in flight; %d rows per batch",
             workers, 2 * workers if workers > 1 else 1, args.batch_size)
    with log_stage("Sort mining rows before training conversion"):
        reader = connection.execute("SELECT m.*, d.finepdf_idx AS positive_idx FROM mined m "
                                    "JOIN doc_mapping d ON m.doc_id_int=d.old_idx "
                                    "ORDER BY d.finepdf_idx,m.doc_id_int").fetch_record_batch(args.batch_size)
    next_query_id, counts, last_report = 0, set(), time.monotonic()
    audit_writer = None
    alias_stats = {"policy": args.negative_alias_policy,
                   "selection": "Highest retrieval score per canonical negative; lowest mining ID breaks ties. Paired scores are retained.",
                   "removed_self_aliases": 0, "removed_duplicate_aliases": 0,
                   "removed_empty_negatives": 0, "removed_empty_skipped": 0,
                   "affected_training_rows": 0, "empty_negative_rows": 0,
                   "candidate_count_before": 0, "candidate_count_after": 0}
    for result in training_batches(reader, query_path, remap, args.batch_size, workers,
                                   args.negative_alias_policy, getattr(args, "excluded_document_ids", ())):
        writer.write(result["train"])
        next_query_id = result["next_query_id"]
        counts.update(result["negative_counts"])
        for counter in COUNTERS:
            alias_stats[counter] += result["counters"][counter]
        if result["audit"] is not None:
            if audit_writer is None:
                audit_writer = ShardedWriter(root, "negative_alias_report", NEGATIVE_ALIAS_SCHEMA, args.rows_per_shard)
                LOG.warning("Removing invalid candidate references; original IDs and paired scores are retained in %s",
                            audit_writer.directory)
            audit_writer.write(result["audit"])
        now = time.monotonic()
        if now - last_report >= 60:
            LOG.info("Training progress: %s grouped rows, %s queries including alternatives",
                     format(writer.rows, ",d"), format(next_query_id, ",d"))
            LOG.info("Candidate aliases removed: %s positive aliases, %s repeated negatives; %s affected rows",
                     format(alias_stats["removed_self_aliases"], ",d"), format(alias_stats["removed_duplicate_aliases"], ",d"),
                     format(alias_stats["affected_training_rows"], ",d"))
            last_report = now
    if next_query_id == 0:
        raise ValueError("Release has no training queries")
    alias_stats["audit"] = audit_writer.close() if audit_writer else None
    args.negative_alias_resolution = alias_stats
    LOG.info("Empty candidate artifacts removed: %s scored negatives, %s skipped slots",
             format(alias_stats["removed_empty_negatives"], ",d"),
             format(alias_stats["removed_empty_skipped"], ",d"))
    LOG.info("Negative alias resolution: %s positive aliases, %s repeated negatives removed across %s rows; "
             "%s rows have no remaining negatives",
             format(alias_stats["removed_self_aliases"], ",d"), format(alias_stats["removed_duplicate_aliases"], ",d"),
             format(alias_stats["affected_training_rows"], ",d"), format(alias_stats["empty_negative_rows"], ",d"))
    return writer.close(), next_query_id, sorted(counts)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dir", type=Path, required=True)
    p.add_argument("--embeddings-dir", type=Path,
                   help="Canonical embedding shards, if stored separately from index-dir")
    p.add_argument("--hard-negatives-dir", type=Path, required=True)
    p.add_argument("--queries-glob", required=True, help="Original or compacted generation Parquets")
    p.add_argument("--exclude-generation-file", type=Path, action="append", default=[],
                   help="Explicitly omit a matched metadata file; repeatable. Every mined query still requires metadata.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--finepdf-root", type=Path, help="Local pinned snapshot root")
    source.add_argument("--source-index-dir", type=Path,
                        help="Portable ID-only source index built where the original FinePDFs shards exist")
    p.add_argument("--finepdf-glob", help="ALL source shards for this config/split (with --finepdf-root)")
    p.add_argument("--source-repo", default="HuggingFaceFW/finepdfs")
    p.add_argument("--source-config", choices=("eng_Latn", "fra_Latn"), required=True)
    p.add_argument("--source-split", default="train")
    p.add_argument("--expected-source-manifest", type=Path, help="Require exact agreement with a published source manifest")
    p.add_argument("--source-revision", required=True, help="Exact 40-character source commit SHA")
    p.add_argument("--provenance-json", type=Path, required=True,
                   help="Models, score semantics, generation/mining settings, and release license")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, help="Parent directory for DuckDB and spill files")
    p.add_argument("--resume-work-dir", type=Path,
                   help="Resume after generation staging or aggregation using saved artifacts read-only")
    p.add_argument("--resume-generation-rows", type=int,
                   help="Expected generation origin count from the completed staging log (required for legacy raw-stage resume)")
    p.add_argument("--resume-generation-parquet", type=Path,
                   help="Reuse a completed generation aggregation from another saved attempt with --resume-work-dir")
    p.add_argument("--mining-cache-dir", type=Path,
                   help="Persistent compaction cache; defaults to mining-cache under the work parent")
    p.add_argument("--keep-work", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--negative-alias-policy", choices=("drop", "error"), default="drop",
                   help="Remove proven canonical self/duplicate aliases with an audit, or stop [drop]")
    p.add_argument("--missing-document-policy", choices=("error", "drop-empty-negatives"), default="error",
                   help="Stop on missing source documents, or exclude only verified empty, queryless counter-ID candidates [error]")
    p.add_argument("--serialized-query-policy", choices=("error", "exclude-groups"), default="error",
                   help="Stop on historically mined serialized byte strings, or exclude their entire query groups [error]")
    p.add_argument("--metadata-fields", default=DEFAULT_METADATA_FIELDS)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--mining-files-per-batch", type=int, default=64,
                   help="Maximum input files compacted by each isolated Arrow worker [64]")
    p.add_argument("--rows-per-shard", type=int, default=100_000)
    p.add_argument("--memory-limit", default="8GB")
    p.add_argument("--generation-memory-limit", default="8GB",
                   help="Separate DuckDB budget for each isolated generation worker [8GB]")
    p.add_argument("--generation-partitions", type=int,
                   help="Generation hash partitions; default: one per 2M origins, capped at 128")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--train-workers", type=int, default=1,
                   help="CPU processes converting training batches; independent of DuckDB --threads [1]")
    args = p.parse_args(argv)
    for name in ("batch_size", "mining_files_per_batch", "rows_per_shard", "threads", "train_workers"):
        if getattr(args, name) <= 0:
            p.error("--{} must be positive".format(name.replace("_", "-")))
    for name in ("generation_partitions", "resume_generation_rows"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            p.error("--{} must be positive".format(name.replace("_", "-")))
    if args.resume_generation_parquet and not args.resume_work_dir:
        p.error("--resume-generation-parquet requires --resume-work-dir")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_revision):
        p.error("--source-revision must be the exact 40-character commit, not main or a version tag")
    args.source_revision = args.source_revision.lower()
    if args.finepdf_root:
        if not args.finepdf_glob:
            p.error("--finepdf-root requires --finepdf-glob")
        args.finepdf_root = args.finepdf_root.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def run(args):
    if args.resume_work_dir:
        from resume_finepdf_release import resume
        return resume(args)
    started = time.monotonic()
    LOG.info("Starting release: %s, config=%s, split=%s, revision=%s",
             args.source_repo, args.source_config, args.source_split, args.source_revision)
    LOG.info("Output: %s; memory limit: %s; threads: %d; rows per shard: %s",
             args.output_dir, args.memory_limit, args.threads, format(args.rows_per_shard, ",d"))
    LOG.info("Index: %s; mining outputs: %s; generation inputs: %s",
             args.index_dir, args.hard_negatives_dir, args.queries_glob)
    LOG.info("Slurm memory per node: %s MiB; per CPU: %s MiB; %s",
             os.environ.get("SLURM_MEM_PER_NODE", "unset"), os.environ.get("SLURM_MEM_PER_CPU", "unset"),
             memory_status())
    LOG.info("Mining files per batch: %d; Arrow rows per batch: %d; DuckDB %s; PyArrow %s",
             args.mining_files_per_batch, args.batch_size, duckdb.__version__, pa.__version__)
    pa.set_cpu_count(args.threads)
    if args.output_dir.exists():
        raise ValueError("Output already exists; choose a new release directory")
    provenance = load_json(args.provenance_json)
    for key in ("license", "generation", "mining", "reranker"):
        if key not in provenance:
            raise ValueError("provenance JSON must include {}".format(key))
    if "REPLACE_WITH" in json.dumps(provenance):
        raise ValueError("Fill every REPLACE_WITH placeholder in the provenance JSON before export")
    if not isinstance(provenance["license"], str) or not provenance["license"]:
        raise ValueError("Provenance license must be a nonempty string")
    for section in ("generation", "mining", "reranker"):
        if not isinstance(provenance[section], dict) or not provenance[section].get("model"):
            raise ValueError("Provenance {} must identify its model".format(section))
    for section in ("mining", "reranker"):
        if not provenance[section].get("score_semantics"):
            raise ValueError("Provenance {} must document score_semantics".format(section))
    # Fail on unusable metadata before canonical scans, compaction, and ID joins.
    generation_files, generation_manifest = generation_inputs(args)
    embeddings_dir = args.embeddings_dir or args.index_dir / "embeddings"
    embeddings = sorted(embeddings_dir.glob("shard_*.parquet"))
    if not embeddings:
        raise ValueError("No canonical embedding shards in index-dir/embeddings")
    LOG.info("Inspecting %d canonical embedding files in %s", len(embeddings), embeddings_dir)
    with log_stage("Read canonical corpus row counts"):
        args.mining_doc_count = sum(pq.ParquetFile(path).metadata.num_rows for path in embeddings)
    LOG.info("Canonical corpus: %s document rows", format(args.mining_doc_count, ",d"))
    if args.finepdf_root and re.fullmatch(r"[0-9a-fA-F]{40}", args.finepdf_root.name):
        if args.finepdf_root.name.lower() != args.source_revision:
            raise ValueError("Local snapshot directory commit differs from --source-revision")
    query_counts = []
    with log_stage("Validate canonical query ordering"):
        for file_number, embedding in enumerate(embeddings, 1):
            count, processed, last_report = 0, 0, time.monotonic()
            pf = pq.ParquetFile(embedding)
            LOG.info("Canonical file %d/%d: %s (%s document rows)",
                     file_number, len(embeddings), embedding, format(pf.metadata.num_rows, ",d"))
            for batch in pf.iter_batches(
                    batch_size=args.batch_size, columns=["all_query_texts", "repr_query_text"]):
                for queries, representative in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                    count += len(queries or [])
                    if queries and representative != queries[0]:
                        raise ValueError("Representative query is not query[0] in {}; exporter cannot "
                                         "label these scores correctly".format(embedding))
                processed += batch.num_rows
                now = time.monotonic()
                if now - last_report >= 60:
                    LOG.info("Canonical file %d/%d: %s/%s documents checked, %s queries",
                             file_number, len(embeddings), format(processed, ",d"),
                             format(pf.metadata.num_rows, ",d"), format(count, ",d"))
                    last_report = now
            query_counts.append(count)
            LOG.info("Canonical file %d/%d finished: %s queries", file_number, len(embeddings), format(count, ",d"))
    LOG.info("Canonical query count: %s", format(sum(query_counts), ",d"))
    with log_stage("Locate completed mining outputs"):
        mining_files, missing = discover_mining(args.hard_negatives_dir, embeddings, args.allow_partial, query_counts)
    if args.source_index_dir:
        from finepdf_release import read_source_manifest
        source = read_source_manifest(args.source_index_dir)
        for field, value in (("repo_id", args.source_repo), ("config", args.source_config),
                             ("split", args.source_split), ("revision", args.source_revision)):
            if source[field] != value:
                raise ValueError("Portable source index {} differs from export settings".format(field))
        if not (args.source_index_dir / "_SUCCESS").is_file():
            raise ValueError("Portable source index is incomplete")
        index_manifest = load_json(args.source_index_dir / "source_index.json")
        sources, actual_rows = [], 0
        for file_number, shard in enumerate(index_manifest["shards"], 1):
            path = args.source_index_dir / shard["path"]
            if Path(shard["path"]).is_absolute() or ".." in Path(shard["path"]).parts:
                raise ValueError("Invalid source-index shard path")
            with log_stage("Verify portable source file {}/{}: {}".format(
                    file_number, len(index_manifest["shards"]), path)):
                if sha256(path) != shard["sha256"]:
                    raise ValueError("Portable source index checksum mismatch: {}".format(path))
            rows = pq.ParquetFile(path).metadata.num_rows
            if rows != shard["num_rows"]:
                raise ValueError("Portable source index shard row count mismatch")
            actual_rows += rows
            sources.append(path.resolve())
        if actual_rows != source["num_rows"] or index_manifest["num_rows"] != source["num_rows"] or not sources:
            raise ValueError("Portable source index row count mismatch")
        entries, source_rows = source["shards"], source["num_rows"]
    else:
        sources = matching_files(args.finepdf_glob)
        LOG.info("Inspecting source snapshot: %s (%d files)", args.finepdf_root, len(sources))
        entries, source_rows = source_manifest(sources, args.finepdf_root)
    if args.expected_source_manifest:
        expected = load_json(args.expected_source_manifest)
        actual = {"repo_id": args.source_repo, "config": args.source_config, "split": args.source_split,
                  "revision": args.source_revision, "num_rows": source_rows, "shards": entries}
        if any(expected.get(key) != value for key, value in actual.items()):
            raise ValueError("Source files/order differ from --expected-source-manifest")
    LOG.info("Source manifest: %s document rows across %d corpus shards",
             format(source_rows, ",d"), len(entries))
    prefix = "data/{}/{}/".format(args.source_config, args.source_split)
    if any(not entry["path"].startswith(prefix) for entry in entries):
        raise ValueError("FinePDFs shard paths must belong to {} under --finepdf-root".format(prefix))
    LOG.info("Inputs ready: %d mining files, %d source files, %d generation files",
             len(mining_files), len(sources), len(generation_files))
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_parent = args.work_dir or args.output_dir.parent
    work_parent.mkdir(parents=True, exist_ok=True)
    args.work = Path(tempfile.mkdtemp(prefix="finepdf-release-work-", dir=work_parent))
    staging = Path(tempfile.mkdtemp(prefix=args.output_dir.name + ".partial-", dir=args.output_dir.parent))
    LOG.info("Work directory: %s; unpublished output: %s", args.work, staging)
    connection = duckdb.connect(str(args.work / "export.duckdb"))
    try:
        connection.execute("SET memory_limit={}".format(sql_literal(args.memory_limit)))
        connection.execute("SET threads={}".format(args.threads))
        connection.execute("SET temp_directory={}".format(sql_literal(args.work / "spill")))
        connection.execute("SET preserve_insertion_order=false")
        with log_stage("Build document ID mapping"):
            remap = build_mapping(connection, args, mining_files, sources, entries)
        if not args.allow_partial:
            with log_stage("Verify mining query coverage"):
                expected_queries = sum(query_counts)
                actual_queries = connection.execute("SELECT sum(len(qids)) FROM mined").fetchone()[0]
                if expected_queries != actual_queries + args.query_group_exclusion["excluded_queries"]:
                    raise ValueError("Mining query coverage mismatch: expected {}, got {}".format(
                        expected_queries, actual_queries))
                LOG.info("Mining query coverage verified: %s queries", format(actual_queries, ",d"))
        raw_generation = args.work / "generation_raw.parquet"
        with log_stage("Stage generation metadata"):
            stage_generation(generation_files, raw_generation,
                             [f.strip() for f in args.metadata_fields.split(",") if f.strip()], args.batch_size,
                             stats=generation_manifest, excluded_query_keys=args.excluded_generation_query_keys)
        write_json(args.work / "generation_staged.json", {
            "raw_file": file_identity(raw_generation),
            "num_rows": pq.ParquetFile(raw_generation).metadata.num_rows,
            "metadata_fields": args.metadata_fields, "generation_inputs": generation_manifest})
        # Ensure the completed ID mapping is durable before the long sort.
        connection.execute("CHECKPOINT")
        generation = args.work / "generation.parquet"
        with log_stage("Sort and aggregate generation origins"):
            aggregate_origins(connection, raw_generation, generation, args.batch_size,
                              partitions=args.generation_partitions, memory_limit=args.generation_memory_limit,
                              stage=log_stage)
        with log_stage("Recover generation UTF-8 query keys"):
            generation, generation_manifest["query_encoding_recovery"] = recover_query_keys(
                connection, generation, args.work, args.batch_size)
        manifest = finish_release(connection, args, remap, generation, staging, generation_manifest,
                                  missing, entries, source_rows, provenance)
        del remap
    except Exception:
        LOG.exception("Export failed; work and unpublished output retained at %s and %s", args.work, staging)
        raise
    finally:
        connection.close()
    if not args.keep_work:
        with log_stage("Remove temporary work directory"):
            shutil.rmtree(args.work)
    else:
        LOG.info("Keeping work directory: %s", args.work)
    LOG.info("Export completed in %.1f min; success marker: %s",
             (time.monotonic() - started) / 60, args.output_dir / "_SUCCESS")
    return manifest


def finish_release(connection, args, remap, generation, staging, generation_manifest, missing, entries, source_rows, provenance):
    with log_stage("Join mining queries to generation metadata"):
        query_path = resolve_query_metadata(connection, args, generation)
    with log_stage("Write document reference shards"):
        refs = ShardedWriter(staging, "document_refs", REF_SCHEMA, args.rows_per_shard)
        reader = connection.execute("SELECT finepdf_idx,finepdf_id,source_shard,row_in_shard "
                                    "FROM source_matches ORDER BY finepdf_idx").fetch_record_batch(args.batch_size)
        for batch in reader:
            refs.write(pa.Table.from_batches([batch]))
        ref_manifest = refs.close()
    with log_stage("Write grouped training shards"):
        train_manifest, queries, negative_counts = export_train(connection, args, remap, query_path, staging)
    del remap
    LOG.info("Writing manifests, dataset card, and helper files")
    identity_report = args.work / "source-identity-report"
    if identity_report.is_dir():
        shutil.copytree(identity_report, staging / "source_identity_report")
        args.source_identity_resolution["report_dir"] = "source_identity_report"
        LOG.info("Copied source identity reports: %s", staging / "source_identity_report")
    source = {"repo_id": args.source_repo, "config": args.source_config, "split": args.source_split,
              "revision": args.source_revision, "num_rows": source_rows, "shards": entries}
    write_json(staging / "source_manifest.json", source)
    manifest = {"format_version": RELEASE_FORMAT_VERSION, "references_only": True,
                "partial": bool(missing) or args.allow_partial, "missing_mining_shards": missing,
                "query_count": queries, "negative_counts": negative_counts,
                "source_identity_resolution": args.source_identity_resolution,
                "negative_alias_resolution": args.negative_alias_resolution,
                "query_group_exclusion": args.query_group_exclusion,
                "generation_inputs": generation_manifest,
                "source_manifest": "source_manifest.json", "provenance": provenance,
                "configs": {"train": train_manifest, "document_refs": ref_manifest}}
    write_json(staging / "release_manifest.json", manifest)
    write_dataset_card(staging, manifest, source)
    LOG.info("Wrote %s", staging / "README.md")
    for name in ("finepdf_release.py", "release_data.py", "materialize_release_training.py"):
        shutil.copyfile(Path(__file__).with_name(name), staging / name)
        LOG.info("Copied helper: %s", staging / name)
    from release_publication import prepare_publication
    internal_dir = args.output_dir.with_name(args.output_dir.name + ".internal-" + uuid.uuid4().hex[:12])
    prepare_publication(staging, internal_dir)
    exclusion_report = args.work / "excluded_query_groups.json"
    if exclusion_report.is_file():
        shutil.copyfile(exclusion_report, internal_dir / exclusion_report.name)
    # The caller receives run diagnostics; the file shipped publicly is simpler.
    manifest["internal_report_dir"] = str(internal_dir)
    LOG.info("Internal reports and export provenance retained at %s", internal_dir)
    write_json(staging / "_SUCCESS", {"format_version": RELEASE_FORMAT_VERSION})
    LOG.info("Publishing completed directory: %s -> %s", staging, args.output_dir)
    os.rename(staging, args.output_dir)
    LOG.info("Release ready: %s (%s grouped rows, %s queries, %s document refs; negative counts %s)",
             args.output_dir, format(train_manifest["num_rows"], ",d"), format(queries, ",d"),
             format(ref_manifest["num_rows"], ",d"), negative_counts)
    return manifest


def write_dataset_card(root, manifest, source):
    from release_publication import render_readme
    (root / "README.md").write_text(render_readme(manifest, source), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
