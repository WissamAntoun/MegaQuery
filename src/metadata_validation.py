#!/usr/bin/env python3
"""Report unmatched generation records from an existing release database.

Reads the saved mapping and compacted mining view without changing the database,
recompacting mining files, or repeating generation aggregation. Query examples
are included in the report; full source document text is never read or written.
"""

import ast
import json
import logging
from pathlib import Path
import unicodedata

import duckdb

from release_data import write_json
from generation_query_encoding import parquet_source


LOG = logging.getLogger(__name__)


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def normalization_hint(query):
    return " ".join(unicodedata.normalize("NFC", query).split())


def missing_metadata(connection, generation_path, report_dir, sample_size=20):
    """Collect exact failures once and classify document absence separately."""
    connection.execute("CREATE OR REPLACE TEMP VIEW diagnostic_generation AS "
                       "SELECT finepdf_id,query FROM {}".format(parquet_source(generation_path)))
    LOG.info("Collecting mined queries with no exact generation record")
    connection.execute("CREATE OR REPLACE TEMP TABLE missing_generation_queries AS "
                       "SELECT q.*,d.finepdf_id,d.finepdf_idx AS positive_idx,d.source_shard,d.row_in_shard "
                       "FROM (SELECT doc_id_int,unnest(query_texts) AS query,unnest(qids) AS mining_query_id, "
                       "generate_subscripts(query_texts,1) AS position FROM mined) q "
                       "JOIN doc_mapping d ON q.doc_id_int=d.old_idx "
                       "ANTI JOIN diagnostic_generation g ON d.finepdf_id=g.finepdf_id AND q.query=g.query")
    count = connection.execute("SELECT count(*) FROM missing_generation_queries").fetchone()[0]
    summary = {"missing_queries": count, "match_policy": "Exact document UUID and query text; no inferred metadata."}
    if not count:
        return summary
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    missing_path = report_dir / "missing_queries.parquet"
    if missing_path.exists():
        raise ValueError("Diagnostic output already exists; choose a new report directory: {}".format(report_dir))
    LOG.info("Found %s missing queries; writing %s", format(count, ",d"), missing_path)
    connection.execute("COPY missing_generation_queries TO {} (FORMAT PARQUET, COMPRESSION ZSTD)".format(
        literal(missing_path)))
    documents, uuids, primary = connection.execute(
        "SELECT count(DISTINCT doc_id_int),count(DISTINCT finepdf_id),count(*) FILTER(WHERE position=1) "
        "FROM missing_generation_queries").fetchone()
    total = connection.execute("SELECT coalesce(sum(len(qids)),0) FROM mined").fetchone()[0]
    summary.update(affected_mining_documents=documents, affected_finepdf_ids=uuids,
                   primary_queries_missing=primary, total_mined_queries=total,
                   fraction_mined_queries_missing=count / total if total else None)
    # Scan only the UUID column and retain IDs relevant to missing queries.
    LOG.info("Checking whether affected document IDs occur anywhere in generation metadata")
    connection.execute("CREATE OR REPLACE TEMP TABLE present_generation_ids AS "
                       "SELECT DISTINCT g.finepdf_id FROM diagnostic_generation g "
                       "SEMI JOIN (SELECT DISTINCT finepdf_id FROM missing_generation_queries) m USING(finepdf_id)")
    absent, present = connection.execute(
        "SELECT count(*) FILTER(WHERE p.finepdf_id IS NULL),count(*) FILTER(WHERE p.finepdf_id IS NOT NULL) "
        "FROM missing_generation_queries m LEFT JOIN present_generation_ids p USING(finepdf_id)").fetchone()
    summary.update(queries_with_document_absent=absent, queries_with_document_present_but_text_missing=present)
    sample_sql = ("SELECT m.* FROM missing_generation_queries m {} JOIN present_generation_ids p "
                  "USING(finepdf_id) ORDER BY mining_query_id LIMIT {}")
    # Include both categories even if one dominates the corpus.
    examples = []
    for kind in ("ANTI", "SEMI"):
        cursor = connection.execute(sample_sql.format(kind, sample_size))
        names = [field[0] for field in cursor.description]
        examples.extend(dict(zip(names, row)) for row in cursor.fetchall())
    candidate_queries = {}
    if examples:
        connection.execute("CREATE OR REPLACE TEMP TABLE sampled_missing_ids(finepdf_id VARCHAR)")
        connection.executemany("INSERT INTO sampled_missing_ids VALUES (?)", [(identifier,) for identifier in
                               sorted({row["finepdf_id"] for row in examples})])
        cursor = connection.execute("SELECT g.finepdf_id,g.query FROM diagnostic_generation g "
                                    "SEMI JOIN sampled_missing_ids s USING(finepdf_id) "
                                    "QUALIFY row_number() OVER(PARTITION BY g.finepdf_id ORDER BY g.query)<=8")
        for identifier, query in cursor.fetchall():
            candidate_queries.setdefault(identifier, []).append(query)
    for example in examples:
        candidates = candidate_queries.get(example["finepdf_id"], [])
        example["generation_query_examples"] = candidates
        example["normalized_text_matches_example"] = any(
            normalization_hint(example["query"]) == normalization_hint(query) for query in candidates)
        example["mined_query_is_serialized_list"] = False
        if example["query"].lstrip().startswith("["):
            try:
                example["mined_query_is_serialized_list"] = isinstance(ast.literal_eval(example["query"]), list)
            except (ValueError, SyntaxError, TypeError, RecursionError):
                pass
    write_json(report_dir / "examples.json", examples)
    write_json(report_dir / "summary.json", summary)
    LOG.error("Generation metadata coverage failed: %s missing queries. Reports: %s",
              format(count, ",d"), report_dir)
    return summary

