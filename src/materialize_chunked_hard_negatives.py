#!/usr/bin/env python3
"""Materialize chunked hard-negative shard outputs into one parquet per shard.

Chunked Pass 2 writes a dataset-like layout:

    output/shard_00012/chunk_00000/*.parquet
    output/shard_00012/chunk_00001/*.parquet

This script materializes each completed shard directory in place by default:

    output/shard_00012/hard_negatives.parquet
    output/shard_00012/_MATERIALIZED_SUCCESS

The marker lets generate_hard_negatives_v4.py treat a materialized shard as
Pass 2 complete on relaunch, even after the original chunk directories are
deleted.
"""

import argparse
import json
import os
import shutil
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_OUTPUT_BASENAME = "hard_negatives.parquet"
MATERIALIZED_MARKER = "_MATERIALIZED_SUCCESS"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Write completed chunked generate_hard_negatives_v4 shard outputs "
            "to one parquet file per shard."
        )
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="Pipeline output directory containing shard_XXXXX directories.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output root for materialized shard directories. Defaults to "
            "--input-dir for in-place materialization."
        ),
    )
    p.add_argument(
        "--output-basename",
        default=DEFAULT_OUTPUT_BASENAME,
        help=f"Final parquet filename inside each shard directory [{DEFAULT_OUTPUT_BASENAME}].",
    )
    p.add_argument(
        "--shards",
        default=None,
        help="Optional comma-separated shard indices to materialize, e.g. 12,13,20.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing materialized parquet and marker.",
    )
    p.add_argument(
        "--allow-incomplete-chunks",
        action="store_true",
        help="Allow chunk directories without a _SUCCESS marker.",
    )
    p.add_argument(
        "--delete-chunks",
        action="store_true",
        help="Delete source chunk_XXXXX directories after successful materialization.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing or deleting files.",
    )
    return p.parse_args()


def _parse_shard_filter(value):
    if not value:
        return None
    shards = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            shards.add(int(item))
    return shards


def _shard_idx_from_name(name):
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _list_shard_dirs(input_dir, shard_filter):
    shard_dirs = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not name.startswith("shard_") or not os.path.isdir(path):
            continue
        shard_idx = _shard_idx_from_name(name)
        if shard_idx is None:
            continue
        if shard_filter is not None and shard_idx not in shard_filter:
            continue
        shard_dirs.append((shard_idx, name, path))
    return shard_dirs


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _parquet_files(path, output_basename):
    if not os.path.isdir(path):
        return []
    excluded = {output_basename}
    return sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.endswith(".parquet")
        and name not in excluded
        and not name.endswith(".tmp")
        and os.path.isfile(os.path.join(path, name))
    )


def _chunk_dirs(shard_dir):
    return sorted(
        os.path.join(shard_dir, name)
        for name in os.listdir(shard_dir)
        if name.startswith("chunk_") and os.path.isdir(os.path.join(shard_dir, name))
    )


def _success_path(chunk_dir):
    return os.path.join(chunk_dir, "_SUCCESS")


def _marker_path(shard_dir):
    return os.path.join(shard_dir, MATERIALIZED_MARKER)


def _is_materialized(shard_dir, output_basename):
    marker = _load_json_if_exists(_marker_path(shard_dir))
    output_path = os.path.join(shard_dir, output_basename)
    return marker is not None and os.path.isfile(output_path)


def _atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _chunk_metadata(shard_dir):
    return _load_json_if_exists(os.path.join(shard_dir, "_chunked_meta.json"))


def _validate_chunks(shard_dir, allow_incomplete_chunks, output_basename):
    chunks = _chunk_dirs(shard_dir)
    if not chunks:
        return [], _parquet_files(shard_dir, output_basename), "flat"

    meta = _chunk_metadata(shard_dir)
    expected = None
    if meta is not None and "num_chunks" in meta:
        expected = int(meta["num_chunks"])
        if len(chunks) != expected:
            raise RuntimeError(
                f"{shard_dir} has {len(chunks)} chunk directories but metadata "
                f"expects {expected}."
            )

    parts = []
    for chunk_idx, chunk_dir in enumerate(chunks):
        if not os.path.exists(_success_path(chunk_dir)) and not allow_incomplete_chunks:
            raise RuntimeError(
                f"Chunk lacks _SUCCESS marker: {chunk_dir}. "
                "Use --allow-incomplete-chunks only if you trust this output."
            )
        chunk_parts = _parquet_files(chunk_dir, output_basename)
        if not chunk_parts:
            raise RuntimeError(f"No parquet files found in chunk {chunk_dir}")
        parts.extend(chunk_parts)

    if expected is not None:
        chunk_ids = sorted(_shard_idx_from_name(os.path.basename(path)) for path in chunks)
        if chunk_ids != list(range(expected)):
            raise RuntimeError(
                f"{shard_dir} chunk directories are not contiguous 0..{expected - 1}."
            )

    return chunks, parts, "chunked"


def _write_single_parquet(part_paths, output_path, dry_run):
    if dry_run:
        print(f"[dry-run] would write {len(part_paths)} parquet parts -> {output_path}")
        return None, None

    tmp_path = output_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    writer = None
    rows = 0
    schema = None
    try:
        for part_path in part_paths:
            pf = pq.ParquetFile(part_path)
            if writer is None:
                schema = pf.schema_arrow
                writer = pq.ParquetWriter(tmp_path, schema)
            for batch in pf.iter_batches():
                table = pa.Table.from_batches([batch])
                table = table.cast(schema)
                writer.write_table(table)
                rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    os.replace(tmp_path, output_path)
    return rows, schema


def materialize_shard(
    shard_name,
    shard_dir,
    out_root,
    output_basename,
    overwrite,
    allow_incomplete_chunks,
    delete_chunks,
    dry_run,
):
    out_shard_dir = os.path.join(out_root, shard_name)
    output_path = os.path.join(out_shard_dir, output_basename)
    marker_path = _marker_path(out_shard_dir)
    in_place = os.path.abspath(out_shard_dir) == os.path.abspath(shard_dir)

    if _is_materialized(out_shard_dir, output_basename) and not overwrite:
        print(f"{shard_name}: already materialized -> {output_path}")
        return "skipped", 0, None

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Materialized parquet exists without a trusted marker: {output_path}. "
            "Use --overwrite if it should be replaced."
        )

    chunks, part_paths, layout = _validate_chunks(
        shard_dir, allow_incomplete_chunks, output_basename
    )
    if not part_paths:
        raise RuntimeError(f"No parquet files found in {shard_dir}")

    if dry_run:
        print(f"[dry-run] would create {out_shard_dir}")
    else:
        os.makedirs(out_shard_dir, exist_ok=True)

    rows, schema = _write_single_parquet(part_paths, output_path, dry_run)
    marker = {
        "source": os.path.abspath(shard_dir),
        "output": os.path.abspath(output_path),
        "layout": layout,
        "chunks": len(chunks),
        "source_parquet_files": len(part_paths),
        "rows": None if rows is None else int(rows),
        "created_at_unix": time.time(),
    }
    if schema is not None:
        marker["schema"] = str(schema)

    if dry_run:
        print(f"[dry-run] would write marker {marker_path}")
    else:
        _atomic_write_json(marker, marker_path)

    if delete_chunks and chunks:
        for chunk_dir in chunks:
            if dry_run:
                print(f"[dry-run] would delete {chunk_dir}")
            else:
                shutil.rmtree(chunk_dir)

    if not in_place and layout == "flat":
        print(
            f"{shard_name}: materialized from flat source into separate output root; "
            "source flat files were left untouched."
        )

    return layout, len(part_paths), rows


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or input_dir)

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(input_dir)

    shard_filter = _parse_shard_filter(args.shards)
    shard_dirs = _list_shard_dirs(input_dir, shard_filter)
    if not shard_dirs:
        print("No shard directories found.", file=sys.stderr)
        return 1

    totals = {"chunked": 0, "flat": 0, "skipped": 0, "files": 0, "rows": 0}
    if args.dry_run:
        print(f"[dry-run] output root: {output_dir}")

    for shard_idx, shard_name, shard_dir in shard_dirs:
        layout, files, rows = materialize_shard(
            shard_name,
            shard_dir,
            output_dir,
            args.output_basename,
            args.overwrite,
            args.allow_incomplete_chunks,
            args.delete_chunks,
            args.dry_run,
        )
        totals[layout] += 1
        totals["files"] += files
        if rows is not None:
            totals["rows"] += rows
        rows_text = "not-computed" if rows is None else str(rows)
        print(
            f"{shard_name}: layout={layout} source_parquet_files={files} "
            f"rows={rows_text} source_shard={shard_idx}"
        )

    rows_total_text = "not-computed" if args.dry_run else str(totals["rows"])
    print(
        "Done: %d materialized, %d skipped, %d source parquet files, %s rows -> %s"
        % (
            totals["chunked"] + totals["flat"],
            totals["skipped"],
            totals["files"],
            rows_total_text,
            output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
