#!/usr/bin/env python3
"""Resolve repeated FinePDFs IDs against exact retained document text."""

from bisect import bisect_right
import hashlib
import json
import logging
import os
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from release_data import write_json


LOG = logging.getLogger(__name__)
SOURCE_HASH_SCHEMA = pa.schema([
    ("finepdf_id", pa.string()), ("finepdf_idx", pa.int64()), ("source_shard", pa.string()),
    ("row_in_shard", pa.int64()), ("text_sha256", pa.string()), ("text_chars", pa.int64()),
    ("extractor", pa.string()),
])
CANONICAL_HASH_SCHEMA = pa.schema([
    ("old_idx", pa.int64()), ("finepdf_id", pa.string()), ("canonical_file", pa.string()),
    ("row_in_file", pa.int64()), ("text_sha256", pa.string()), ("text_chars", pa.int64()),
])
RESOLVED_SCHEMA = pa.schema([
    ("old_idx", pa.int64()), ("finepdf_id", pa.string()), ("finepdf_idx", pa.int64()),
    ("source_shard", pa.string()), ("row_in_shard", pa.int64()),
])


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def fingerprint_rows(path, requests, id_column, text_column):
    """Stream selected row groups, decoding only requested text into Python.

    Requests are (row_in_file, expected_id, caller_record). Streaming also bounds
    memory when canonical embedding files have very large text row groups.
    """
    requests = sorted(requests, key=lambda item: item[0])
    with pq.ParquetFile(path) as source:
        names = set(source.schema_arrow.names)
        if not {id_column, text_column} <= names:
            raise ValueError("Full document identity/text missing from {}".format(path))
        starts, offset = [], 0
        for group in range(source.metadata.num_row_groups):
            starts.append(offset)
            offset += source.metadata.row_group(group).num_rows
        grouped = {}
        for row, expected_id, record in requests:
            if row < 0 or row >= offset:
                raise ValueError("Document row outside {}: {}".format(path, row))
            grouped.setdefault(bisect_right(starts, row) - 1, []).append((row, expected_id, record))
        extractor = next((name for name in ("extractor", "processor") if name in names), None)
        columns = [id_column, text_column] + ([extractor] if extractor else [])
        for group, items in sorted(grouped.items()):
            position, base = 0, starts[group]
            for batch in source.iter_batches(batch_size=512, row_groups=[group], columns=columns, use_threads=False):
                end = base + batch.num_rows
                while position < len(items) and items[position][0] < end:
                    row, expected_id, record = items[position]
                    local = row - base
                    actual_id = batch.column(id_column)[local].as_py()
                    if actual_id != expected_id:
                        raise ValueError("Document ID at {} row {} differs from the saved mapping".format(path, row))
                    text = batch.column(text_column)[local].as_py()
                    extractor_value = batch.column(extractor)[local].as_py() if extractor else None
                    yield record, {
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None,
                        "text_chars": len(text) if text is not None else None,
                        "extractor": str(extractor_value) if extractor_value is not None else None,
                    }
                    position += 1
                base = end
                if position == len(items):
                    break
            if position != len(items):
                raise ValueError("Could not read every requested row in {}".format(path))


def write_hashes(path, rows, schema):
    # Only ambiguous IDs are staged; document text is never stored in a report.
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
    LOG.info("Wrote %s (%s fingerprints)", path, format(len(rows), ",d"))


def canonical_fingerprints(records, index_dir, embeddings_dir):
    texts = sorted((Path(index_dir) / "doc_texts").glob("shard_*.parquet")) if index_dir else []
    if texts:
        files, id_column, text_column, kind = texts, "doc_id_int", "text", "indexed doc_texts"
    else:
        files = sorted(Path(embeddings_dir).glob("shard_*.parquet")) if embeddings_dir else []
        id_column, text_column, kind = "doc_id_str", "pos_text", "canonical embedding pos_text"
    if not files:
        raise ValueError("Ambiguous source IDs require full canonical text in index-dir/doc_texts or embeddings-dir")
    starts, offset = [], 0
    for path in files:
        starts.append(offset)
        with pq.ParquetFile(path) as source:
            offset += source.metadata.num_rows
    requests = {}
    for record in records:
        old_idx = record["old_idx"]
        if old_idx < 0 or old_idx >= offset:
            raise ValueError("Mining ID outside canonical text corpus: {}".format(old_idx))
        file_number = bisect_right(starts, old_idx) - 1
        row = old_idx - starts[file_number]
        expected = old_idx if id_column == "doc_id_int" else record["finepdf_id"]
        requests.setdefault(file_number, []).append((row, expected, record))
    result = []
    for number, (file_number, items) in enumerate(sorted(requests.items()), 1):
        path = files[file_number]
        LOG.info("Fingerprint canonical %s file %d/%d: %s (%d selected documents)",
                 kind, number, len(requests), path, len(items))
        for record, fingerprint in fingerprint_rows(path, items, id_column, text_column):
            # Public audit label: identify the artifact without a machine locator.
            namespace = "doc_texts" if texts else "embeddings"
            result.append({**record, "canonical_file": namespace + "/" + path.name,
                           "row_in_file": record["old_idx"] - starts[file_number],
                           "text_sha256": fingerprint["text_sha256"], "text_chars": fingerprint["text_chars"]})
    return result, kind


def resolve_ambiguous_ids(connection, finepdf_root, index_dir, embeddings_dir, report_dir):
    """Return one source row per ambiguous mining ID, or fail with a report."""
    connection.execute("CREATE OR REPLACE TEMP TABLE duplicate_source_ids AS "
                       "SELECT finepdf_id,count(*) AS occurrences FROM source_matches "
                       "GROUP BY finepdf_id HAVING count(*)>1")
    duplicate_ids, extra_rows = connection.execute(
        "SELECT count(*),coalesce(sum(occurrences-1),0) FROM duplicate_source_ids").fetchone()
    connection.execute("CREATE OR REPLACE TEMP TABLE duplicate_mining_ids AS "
                       "SELECT finepdf_id,count(*) AS occurrences FROM mining_ids "
                       "GROUP BY finepdf_id HAVING count(*)>1")
    duplicate_mining_ids = connection.execute("SELECT count(*) FROM duplicate_mining_ids").fetchone()[0]
    connection.execute("CREATE OR REPLACE TEMP TABLE ambiguous_source_ids AS "
                       "SELECT finepdf_id FROM duplicate_source_ids UNION SELECT finepdf_id FROM duplicate_mining_ids")
    ambiguous_ids = connection.execute("SELECT count(*) FROM ambiguous_source_ids").fetchone()[0]
    summary = {"duplicate_source_ids": duplicate_ids, "extra_source_rows": extra_rows,
               "duplicate_mining_uuid_ids": duplicate_mining_ids, "ambiguous_uuid_ids": ambiguous_ids,
               "policy": "One-to-one UUIDs map directly; UUIDs repeated in source or mining require exact canonical text SHA256. "
                         "Identical matching text uses the smallest upstream row index."}
    if not ambiguous_ids:
        return pa.Table.from_pylist([], schema=RESOLVED_SCHEMA), summary
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    LOG.warning("Found %s repeated source IDs (%s extra occurrences), %s repeated mining UUIDs; "
                "checking %s ambiguous UUIDs against exact text. Reports: %s",
                format(duplicate_ids, ",d"), format(extra_rows, ",d"), format(duplicate_mining_ids, ",d"),
                format(ambiguous_ids, ",d"), report_dir)
    sources = connection.execute("SELECT s.* FROM source_matches s JOIN ambiguous_source_ids d USING(finepdf_id) "
                                 "ORDER BY s.source_shard,s.row_in_shard").fetch_arrow_table()
    mining = connection.execute("SELECT m.* FROM mining_ids m JOIN ambiguous_source_ids d USING(finepdf_id) "
                                "ORDER BY m.old_idx").fetch_arrow_table()
    pq.write_table(sources, report_dir / "source_duplicate_rows.parquet", compression="zstd")
    pq.write_table(mining, report_dir / "mining_duplicate_ids.parquet", compression="zstd")
    write_json(report_dir / "summary.json", summary)
    if finepdf_root is None:
        raise ValueError("Ambiguous UUIDs require local FinePDFs text; export with --finepdf-root "
                         "instead of the ID-only portable index. Reports: {}".format(report_dir))
    requests = {}
    for record in sources.to_pylist():
        shard = Path(record["source_shard"])
        if shard.is_absolute() or ".." in shard.parts:
            raise ValueError("Invalid source shard path in duplicate report")
        requests.setdefault(record["source_shard"], []).append((record["row_in_shard"], record["finepdf_id"], record))
    source_hashes = []
    for number, (relative, items) in enumerate(sorted(requests.items()), 1):
        LOG.info("Fingerprint source file %d/%d: %s (%d selected rows)", number, len(requests), relative, len(items))
        for record, fingerprint in fingerprint_rows(Path(finepdf_root) / relative, items, "id", "text"):
            source_hashes.append({**record, **fingerprint})
    write_hashes(report_dir / "source_fingerprints.parquet", source_hashes, SOURCE_HASH_SCHEMA)
    canonical_hashes, kind = canonical_fingerprints(mining.to_pylist(), index_dir, embeddings_dir)
    write_hashes(report_dir / "canonical_fingerprints.parquet", canonical_hashes, CANONICAL_HASH_SCHEMA)
    summary["canonical_text_source"] = kind
    matches, variants = {}, {}
    for record in source_hashes:
        variants.setdefault(record["finepdf_id"], set()).add(record["text_sha256"])
        if record["text_sha256"] is not None:
            matches.setdefault((record["finepdf_id"], record["text_sha256"]), []).append(record)
    resolved, decisions = [], []
    for record in canonical_hashes:
        candidates = matches.get((record["finepdf_id"], record["text_sha256"]), [])
        chosen = min(candidates, key=lambda item: item["finepdf_idx"]) if candidates else None
        decision = {**record, "matching_source_rows": len(candidates),
                    "selected_finepdf_idx": chosen["finepdf_idx"] if chosen else None,
                    "status": "resolved" if chosen else "no_exact_text_match"}
        decisions.append(decision)
        if chosen:
            resolved.append({"old_idx": record["old_idx"], **{name: chosen[name] for name in RESOLVED_SCHEMA.names[1:]}})
    summary.update({"ambiguous_mining_documents": len(canonical_hashes), "resolved_mining_documents": len(resolved),
                    "unmatched_mining_documents": len(canonical_hashes) - len(resolved),
                    "source_ids_with_identical_text": sum(len(values)==1 and None not in values for values in variants.values()),
                    "source_ids_with_different_text": sum(len(values)>1 for values in variants.values()),
                    "identical_match_ties": sum(row["matching_source_rows"]>1 for row in decisions)})
    decision_schema = pa.schema(list(CANONICAL_HASH_SCHEMA) + [
        ("matching_source_rows", pa.int64()), ("selected_finepdf_idx", pa.int64()), ("status", pa.string())])
    pq.write_table(pa.Table.from_pylist(decisions, schema=decision_schema), report_dir / "resolution.parquet", compression="zstd")
    write_json(report_dir / "summary.json", summary)
    if summary["unmatched_mining_documents"]:
        raise ValueError("{} ambiguous mining documents have no exact text match in the source snapshot; "
                         "no source row was selected for them. Inspect {}".format(summary["unmatched_mining_documents"], report_dir))
    LOG.info("Resolved %s ambiguous mining documents; %s identical-content ties. Reports: %s",
             format(len(resolved), ",d"), format(summary["identical_match_ties"], ",d"), report_dir)
    return pa.Table.from_pylist(resolved, schema=RESOLVED_SCHEMA), summary

