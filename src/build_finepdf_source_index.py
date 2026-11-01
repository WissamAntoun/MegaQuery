#!/usr/bin/env python3
"""Build an ID/shard/row index without copying FinePDFs document text."""

import argparse
import glob
import os
from pathlib import Path
import re
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from release_data import REF_SCHEMA, ShardedWriter, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finepdf-root", type=Path, required=True)
    p.add_argument("--finepdf-glob", required=True)
    p.add_argument("--source-repo", default="HuggingFaceFW/finepdfs")
    p.add_argument("--source-config", choices=("eng_Latn", "fra_Latn"), required=True)
    p.add_argument("--source-split", default="train")
    p.add_argument("--source-revision", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rows-per-shard", type=int, default=1_000_000)
    args = p.parse_args(argv)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_revision):
        p.error("--source-revision must be an exact commit SHA")
    if args.rows_per_shard <= 0 or args.output_dir.exists():
        p.error("Shard size must be positive and output directory must not exist")
    files = sorted(Path(f).resolve() for f in glob.glob(args.finepdf_glob, recursive=True) if Path(f).is_file())
    if not files:
        p.error("No source Parquets found")
    root = args.finepdf_root.resolve()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=args.output_dir.name + ".partial-", dir=args.output_dir.parent))
    writer = ShardedWriter(staging, "source_index", REF_SCHEMA, args.rows_per_shard)
    entries, offset = [], 0
    prefix = "data/{}/{}/".format(args.source_config, args.source_split)
    for path in files:
        relative = path.relative_to(root).as_posix()
        if not relative.startswith(prefix):
            raise ValueError("Source shard outside the declared config/split: {}".format(relative))
        pf = pq.ParquetFile(path)
        entries.append({"path": relative, "start_row": offset, "num_rows": pf.metadata.num_rows})
        local_row = 0
        for batch in pf.iter_batches(batch_size=8192, columns=["id"]):
            ids = batch.column(0)
            if ids.null_count:
                raise ValueError("Null source ID in {}".format(path))
            count = batch.num_rows
            table = pa.Table.from_arrays([
                pa.array(range(offset + local_row, offset + local_row + count), type=pa.int64()),
                ids, pa.array([relative] * count),
                pa.array(range(local_row, local_row + count), type=pa.int64()),
            ], schema=REF_SCHEMA)
            writer.write(table)
            local_row += count
        offset += pf.metadata.num_rows
    manifest = writer.close()
    write_json(staging / "source_manifest.json", {
        "repo_id": args.source_repo, "config": args.source_config, "split": args.source_split,
        "revision": args.source_revision.lower(), "num_rows": offset, "shards": entries})
    write_json(staging / "source_index.json", manifest)
    write_json(staging / "_SUCCESS", {"num_rows": offset})
    os.rename(staging, args.output_dir)
    print("Wrote {} source IDs (no document text) to {}".format(offset, args.output_dir))


if __name__ == "__main__":
    main()
