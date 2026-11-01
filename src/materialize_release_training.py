#!/usr/bin/env python3
"""Build a ready-to-train text view from a local references-only release."""

import argparse
import itertools
import os
from pathlib import Path
import tempfile

import pyarrow as pa

from finepdf_release import FinePDFResolver, iter_release_rows, iter_training_examples
from release_data import ShardedWriter, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-dir", type=Path, required=True)
    p.add_argument("--config", help="Language config, e.g. eng_Latn or fra_Latn; required at the combined root")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--local-source-root", type=Path)
    p.add_argument("--cache-dir")
    p.add_argument("--num-negatives", type=int, default=8)
    p.add_argument("--all-queries", action="store_true")
    p.add_argument("--limit", type=int, help="Maximum emitted examples, including additional queries")
    p.add_argument("--rows-per-shard", type=int, default=100_000)
    args = p.parse_args(argv)
    if args.num_negatives < 0 or args.rows_per_shard <= 0 or (args.limit is not None and args.limit <= 0):
        p.error("Invalid negative count, shard size, or limit")
    if args.output_dir.exists():
        p.error("Output already exists; choose a new directory")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=args.output_dir.name + ".partial-", dir=args.output_dir.parent))
    schema = pa.schema([(name, pa.string()) for name in
                        ["query", "positive"] + ["negative_{}".format(i) for i in range(args.num_negatives)]])
    writer = ShardedWriter(staging, "train", schema, args.rows_per_shard)
    with FinePDFResolver(args.release_dir, args.local_source_root, args.cache_dir, config=args.config) as resolver:
        examples = iter_training_examples(iter_release_rows(args.release_dir, config=args.config), resolver,
                                          args.num_negatives, args.all_queries)
        if args.limit is not None:
            examples = itertools.islice(examples, args.limit)
        while True:
            batch = list(itertools.islice(examples, 256))
            if not batch:
                break
            writer.write(pa.Table.from_pylist(batch, schema=schema))
    manifest = writer.close()
    if manifest["num_rows"] == 0:
        raise ValueError("No examples to materialize")
    write_json(staging / "manifest.json", {"train": manifest, "num_negatives": args.num_negatives,
                                          "all_queries": args.all_queries})
    write_json(staging / "_SUCCESS", {"num_rows": manifest["num_rows"]})
    os.rename(staging, args.output_dir)
    print("Wrote {} examples to {}".format(manifest["num_rows"], args.output_dir))


if __name__ == "__main__":
    main()

