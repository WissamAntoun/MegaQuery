"""Finish a release after generation staging, using saved artifacts read-only."""

import json
import logging
from pathlib import Path
import re
import shutil
import tempfile

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from compact_release_mining import file_identity
from generation_query_encoding import recover_query_keys


LOG = logging.getLogger(__name__)


def validate_saved_mining(connection, root):
    sql = connection.execute("SELECT sql FROM duckdb_views() WHERE view_name IN ('mined_all','mined') "
                             "ORDER BY CASE WHEN view_name='mined_all' THEN 0 ELSE 1 END LIMIT 1").fetchone()
    if not sql:
        raise ValueError("Saved mining view is absent; a metadata-stage failure is required")
    paths = [Path(value.replace("''", "'")) for value in re.findall(r"'((?:[^']|'')*)'", sql[0])
             if value.endswith(".parquet")]
    if not paths:
        raise ValueError("Saved mining view has no identifiable compacted files")
    root = Path(root).resolve()
    for path in paths:
        marker = path.parent / "_SUCCESS.json"
        if not marker.is_file():
            raise ValueError("Saved mining cache has no completion metadata: {}".format(path))
        info = json.loads(marker.read_text())
        stat = path.stat()
        if stat.st_size != info["size"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise ValueError("Saved mining cache changed: {}".format(path))
        for record in info["inputs"]:
            source = Path(record["path"]).resolve()
            if root not in source.parents or file_identity(source) != record:
                raise ValueError("Mining input changed or belongs to another directory: {}".format(source))
    LOG.info("Validated %d saved compacted mining files and their input identities", len(paths))


def resume(args):
    from export_finepdf_release import (aggregate_origins, excluded_empty_documents, filter_serialized_query_groups,
                                       finish_release, load_json, log_stage,
                                       matching_files, parquet_sql, require_zero, source_manifest, sql_literal)
    if args.expected_source_manifest:
        raise ValueError("Resume with --expected-source-manifest is unsupported; use a fresh export for pinned-order verification")
    old = args.resume_work_dir.resolve()
    database = old / "export.duckdb"
    generation = (args.resume_generation_parquet.resolve() if args.resume_generation_parquet
                  else old / "generation.parquet")
    raw_generation = old / "generation_raw.parquet"
    map_path = old / "old_to_source.int64"
    if not all(path.is_file() for path in (database, map_path)):
        raise ValueError("Resume requires saved export.duckdb and old_to_source.int64")
    if args.resume_generation_parquet and not generation.is_file():
        raise ValueError("Requested saved generation aggregation does not exist: {}".format(generation))
    aggregate_saved_raw = not generation.is_file()
    if generation.is_file():
        try:
            with pq.ParquetFile(generation) as saved:
                saved.metadata.num_rows
                if saved.schema_arrow.names != ["finepdf_id", "query", "generation_metadata"]:
                    raise ValueError("Saved generation aggregation has an unexpected schema")
        except (pa.ArrowInvalid, OSError):
            if args.resume_generation_parquet:
                raise ValueError("Requested saved generation aggregation is unreadable: {}".format(generation))
            LOG.warning("Saved generation aggregation has no readable footer; checking the completed raw stage")
            aggregate_saved_raw = True
    stage_marker = old / "generation_staged.json"
    stage_info = load_json(stage_marker) if stage_marker.is_file() else None
    if aggregate_saved_raw:
        if not raw_generation.is_file():
            raise ValueError("Resume requires generation.parquet or completed generation_raw.parquet")
        with pq.ParquetFile(raw_generation) as saved:
            raw_rows = saved.metadata.num_rows
            if saved.schema_arrow.names != ["finepdf_id", "query", "origin_json", "sequence"]:
                raise ValueError("Saved raw generation schema differs from the staging schema")
        if stage_info:
            if file_identity(raw_generation) != stage_info["raw_file"]:
                raise ValueError("Saved raw generation file changed after staging")
            if args.metadata_fields != stage_info["metadata_fields"]:
                raise ValueError("Requested metadata fields differ from the saved raw generation stage")
            expected_rows = stage_info["num_rows"]
        else:
            expected_rows = args.resume_generation_rows
            if expected_rows is None:
                raise ValueError("Legacy raw-stage resume requires --resume-generation-rows with the origin count "
                                 "from the completed 'Staged ... generation query origins' log; a readable footer "
                                 "alone does not prove staging finished")
        if raw_rows != expected_rows or (args.resume_generation_rows is not None and raw_rows != args.resume_generation_rows):
            raise ValueError("Saved raw generation row count differs from the completed staging count")
    if args.output_dir.exists():
        raise ValueError("Output already exists; choose a new release directory")
    if not args.finepdf_root:
        raise ValueError("Metadata-stage resume currently requires the local FinePDFs source snapshot")
    if args.allow_partial:
        raise ValueError("Metadata-stage resume requires a full mining run; use a fresh export for a partial release")
    provenance = load_json(args.provenance_json)
    if "REPLACE_WITH" in json.dumps(provenance) or not isinstance(provenance.get("license"), str) or not provenance["license"]:
        raise ValueError("Fill release provenance and license before resuming")
    for section in ("generation", "mining", "reranker"):
        if not isinstance(provenance.get(section), dict) or not provenance[section].get("model"):
            raise ValueError("Provenance {} must identify its model".format(section))
    for section in ("mining", "reranker"):
        if not provenance[section].get("score_semantics"):
            raise ValueError("Provenance {} must document score_semantics".format(section))
    if re.fullmatch(r"[0-9a-fA-F]{40}", args.finepdf_root.name) and args.finepdf_root.name.lower() != args.source_revision:
        raise ValueError("Local source snapshot commit differs from --source-revision")
    saved_generation = raw_generation if aggregate_saved_raw else generation
    generation_stat = saved_generation.stat()
    generation_manifest = {"resumed_from": str(old),
                           "source": "saved_raw_generation" if aggregate_saved_raw else "saved_aggregation",
                           "requested_queries_glob": args.queries_glob,
                           "saved_raw_generation" if aggregate_saved_raw else "saved_aggregation": {
                                                 "path": str(saved_generation), "size_bytes": generation_stat.st_size,
                                                 "mtime_ns": generation_stat.st_mtime_ns,
                                                 "num_rows": pq.ParquetFile(saved_generation).metadata.num_rows},
                           "all_generation_inputs_included": None,
                           "notes": "Legacy saved aggregation does not record original file inventory or exclusions."}
    if stage_info:
        generation_manifest.update(stage_info["generation_inputs"])
        generation_manifest.pop("notes", None)
    sources = matching_files(args.finepdf_glob)
    entries, source_rows = source_manifest(sources, args.finepdf_root)
    prefix = "data/{}/{}/".format(args.source_config, args.source_split)
    if any(not entry["path"].startswith(prefix) for entry in entries):
        raise ValueError("Source shards differ from the resumed config/split")
    parent = args.work_dir or args.output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.work = Path(tempfile.mkdtemp(prefix="finepdf-release-work-", dir=parent))
    staging = Path(tempfile.mkdtemp(prefix=args.output_dir.name + ".partial-", dir=args.output_dir.parent))
    LOG.info("Resuming saved mapping and %s from %s; new work: %s",
             "raw generation origins" if aggregate_saved_raw else "generation aggregation", old, args.work)
    connection = duckdb.connect(str(database), read_only=True)
    try:
        connection.execute("SET memory_limit={}".format(sql_literal(args.memory_limit)))
        connection.execute("SET threads={}".format(args.threads))
        connection.execute("SET temp_directory={}".format(sql_literal(args.work / "spill")))
        connection.execute("SET preserve_insertion_order=false")
        with log_stage("Validate saved mining cache"):
            validate_saved_mining(connection, args.hard_negatives_dir)
        with log_stage("Validate serialized mining queries"):
            filter_serialized_query_groups(connection, args)
        with log_stage("Validate resumed full mining query coverage"):
            embeddings = sorted((args.embeddings_dir or args.index_dir / "embeddings").glob("shard_*.parquet"))
            if not embeddings:
                raise ValueError("Canonical embedding shards are required to validate resumed query coverage")
            args.mining_doc_count = sum(pq.ParquetFile(path).metadata.num_rows for path in embeddings)
            expected = connection.execute("SELECT coalesce(sum(len(all_query_texts)),0) FROM {}".format(
                parquet_sql(embeddings))).fetchone()[0]
            actual = connection.execute("SELECT coalesce(sum(len(qids)),0) FROM mined").fetchone()[0]
            excluded = args.query_group_exclusion["excluded_queries"]
            if expected != actual + excluded:
                raise ValueError("Resumed mining query coverage mismatch: expected {}, got {}".format(expected, actual))
            LOG.info("Resumed query coverage verified: %s queries", format(actual, ",d"))
        with log_stage("Validate resumed index IDs"):
            require_zero(connection, "SELECT count(*) FROM doc_mapping d LEFT JOIN {} m "
                         "ON d.old_idx=m.doc_id_int WHERE m.doc_id_int IS NULL OR d.finepdf_id<>m.doc_id_str".format(
                             parquet_sql([args.index_dir / "doc_id_map.parquet"])), "Saved mapping differs from current mining index")
        connection.register("resume_sources", pa.Table.from_pylist([
            {"filename": str(path), "source_shard": entry["path"], "start_row": entry["start_row"],
             "num_rows": entry["num_rows"]} for path, entry in zip(sources, entries)]))
        require_zero(connection, "SELECT count(*) FROM doc_mapping d LEFT JOIN resume_sources s USING(source_shard) "
                     "WHERE s.source_shard IS NULL OR d.row_in_shard<0 OR d.row_in_shard>=s.num_rows "
                     "OR d.finepdf_idx<>s.start_row+d.row_in_shard", "Saved source coordinates differ from current snapshot")
        with log_stage("Validate resumed source IDs"):
            source_sql = "read_parquet([{}],filename=true,file_row_number=true)".format(
                ",".join(sql_literal(path) for path in sources))
            require_zero(connection, "SELECT count(*) FROM doc_mapping d LEFT JOIN (SELECT s.id,f.source_shard, "
                         "s.file_row_number FROM {} s JOIN resume_sources f ON s.filename=f.filename) c "
                         "ON d.source_shard=c.source_shard AND d.row_in_shard=c.file_row_number "
                         "WHERE c.id IS NULL OR c.id<>d.finepdf_id".format(source_sql),
                         "Saved source IDs differ from current snapshot")
        maximum = connection.execute("SELECT max(old_idx) FROM needed").fetchone()[0]
        if maximum is None or maximum >= args.mining_doc_count or map_path.stat().st_size != (maximum + 1) * 8:
            raise ValueError("Saved integer remapping size is inconsistent")
        remap = np.memmap(map_path, mode="r", dtype="int64", shape=(maximum + 1,))
        with log_stage("Validate saved integer remapping"):
            reader = connection.execute("SELECT old_idx,finepdf_idx FROM doc_mapping").fetch_record_batch(args.batch_size)
            for batch in reader:
                if not np.array_equal(remap[batch.column(0).to_numpy()], batch.column(1).to_numpy()):
                    raise ValueError("Saved integer remapping differs from saved mapping table")
        identity_report = old / "source-identity-report"
        args.source_identity_resolution = load_json(identity_report / "summary.json") if (identity_report / "summary.json").is_file() else {
            "policy": "Resumed saved document mapping; source IDs and coordinates revalidated."}
        args.source_identity_resolution["aliased_mining_documents"] = connection.execute(
            "SELECT count(*)-count(DISTINCT finepdf_idx) FROM doc_mapping").fetchone()[0]
        if identity_report.is_dir():
            shutil.copytree(identity_report, args.work / "source-identity-report")
        with log_stage("Validate missing candidate documents"):
            args.excluded_document_ids = excluded_empty_documents(connection, args)
        if aggregate_saved_raw:
            generation = args.work / "generation.parquet"
            with log_stage("Sort and aggregate saved generation origins"):
                aggregate_origins(connection, raw_generation, generation, args.batch_size,
                                  partitions=args.generation_partitions, memory_limit=args.generation_memory_limit,
                                  stage=log_stage)
        with log_stage("Recover generation UTF-8 query keys"):
            generation_paths, generation_manifest["query_encoding_recovery"] = recover_query_keys(
                connection, generation, args.work, args.batch_size)
        if (saved_generation.stat().st_size != generation_stat.st_size
                or saved_generation.stat().st_mtime_ns != generation_stat.st_mtime_ns):
            raise ValueError("Saved generation artifact changed during resume")
        manifest = finish_release(connection, args, remap, generation_paths, staging, generation_manifest,
                                  [], entries, source_rows, provenance)
        del remap
    except Exception:
        LOG.exception("Resume failed; original saved work is unchanged. New work/output: %s / %s", args.work, staging)
        raise
    finally:
        connection.close()
    if not args.keep_work:
        shutil.rmtree(args.work)
    return manifest
