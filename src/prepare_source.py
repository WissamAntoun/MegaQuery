#!/usr/bin/env python3
"""Download or validate FinePDFs files in a released source manifest's order."""
import argparse
from pathlib import Path
import re

import pyarrow.parquet as pq

from finepdf_release import read_source_manifest


def prepare(release_dir, config, output_dir, local_only=False):
    source = read_source_manifest(release_dir, config=config)
    if config not in ('eng_Latn', 'fra_Latn') or source['config'] != config:
        raise ValueError('Expected eng_Latn or fra_Latn')
    if not re.fullmatch(r'[0-9a-f]{40}', source['revision']):
        raise ValueError('Source revision must be an immutable commit')
    output_dir = Path(output_dir)
    for shard in source['shards']:
        relative = shard['path']
        if not relative.startswith(f'data/{config}/train/'):
            raise ValueError('Shard does not belong to the selected language/split')
        path = output_dir / relative
        if not local_only:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id=source['repo_id'], repo_type='dataset',
                                       revision=source['revision'], filename=relative,
                                       local_dir=str(output_dir)))
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows != shard['num_rows']:
                raise ValueError(f'Source row count mismatch: {relative}')
            if not {'id', 'text'} <= set(parquet.schema_arrow.names):
                raise ValueError(f'Source requires id and text: {relative}')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--config', choices=('eng_Latn', 'fra_Latn'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--local-only', action='store_true', help='Validate existing files without downloading')
    args = parser.parse_args()
    source = prepare(args.release_dir, args.config, args.output_dir, args.local_only)
    print(f"Prepared {len(source['shards'])} files, {source['num_rows']} rows at {source['revision']}")


if __name__ == '__main__':
    main()
