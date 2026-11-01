#!/usr/bin/env python3
"""Copy complete English/French exports into one language-local dataset layout.

All source directories are read-only. Only manifest-listed data files and public
reader/documentation files are copied. The output is installed atomically.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

from finepdf_release import read_source_manifest
from release_data import RELEASE_FORMAT_VERSION, sha256, write_json
from release_publication import public_manifest

CONFIGS = ('eng_Latn', 'fra_Latn')


def checked_path(root, relative):
    value = Path(relative)
    path = root / value
    if value.is_absolute() or '..' in value.parts or root.resolve() not in path.resolve().parents:
        raise ValueError(f'Invalid inventory path: {relative}')
    if not path.is_file():
        raise ValueError(f'Missing inventory file: {relative}')
    return path


def combine(english, french, output):
    output = Path(output).resolve()
    inputs = [Path(english).resolve(), Path(french).resolve()]
    if output.exists() or any(root == output or root in output.parents for root in inputs):
        raise ValueError('Choose a new output directory outside the input releases')
    manifests = []
    licenses = set()
    for config, root in zip(CONFIGS, inputs):
        if not (root / '_SUCCESS').is_file():
            raise ValueError(f'Incomplete release: {config}')
        manifest = json.loads((root / 'release_manifest.json').read_text())
        if manifest.get('format_version') != RELEASE_FORMAT_VERSION or manifest.get('partial'):
            raise ValueError('Expected a complete format 2 release')
        if not manifest.get('references_only') or manifest.get('missing_mining_shards'):
            raise ValueError('Expected a complete references-only release')
        source = read_source_manifest(root, config=config)
        if source['config'] != config:
            raise ValueError('Source language does not match input position')
        for section in ('train', 'document_refs'):
            inventory = manifest['configs'][section]
            offset = 0
            paths = set()
            for shard in inventory['shards']:
                path = checked_path(root, shard['path'])
                if not shard['path'].startswith(section + '/') or shard['path'] in paths:
                    raise ValueError('Invalid or repeated shard path')
                if shard['start_row'] != offset or shard['num_rows'] <= 0:
                    raise ValueError('Invalid shard offsets')
                if sha256(path) != shard['sha256']:
                    raise ValueError(f"Checksum mismatch: {shard['path']}")
                import pyarrow.parquet as pq
                if pq.read_metadata(path).num_rows != shard['num_rows']:
                    raise ValueError('Inventory row count mismatch')
                offset += shard['num_rows']
                paths.add(shard['path'])
            if offset != inventory['num_rows']:
                raise ValueError('Inventory total mismatch')
        license_name = manifest.get('provenance', {}).get('license')
        if not isinstance(license_name, str) or not license_name:
            raise ValueError('Each release must declare its data license')
        licenses.add(license_name)
        manifests.append((public_manifest(manifest), source))
    if len(licenses) != 1:
        raise ValueError('Language releases must use the same data license')
    data_license = licenses.pop()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    try:
        combined = {'format_version': RELEASE_FORMAT_VERSION, 'layout': 'multilingual',
                    'references_only': True, 'partial': False, 'missing_mining_shards': [],
                    'query_count': 0, 'negative_counts': [], 'language_configs': {}, 'configs': {}}
        sources = {'layout': 'multilingual', 'language_configs': {}}
        for config, root, (manifest, source) in zip(CONFIGS, inputs, manifests):
            relative = 'data/' + config
            target = staging / relative
            target.mkdir(parents=True)
            combined['language_configs'][config] = {'path': relative}
            sources['language_configs'][config] = {'path': relative + '/source_manifest.json'}
            combined['query_count'] += manifest['query_count']
            combined['negative_counts'].extend(manifest['negative_counts'])
            for section, inventory in manifest['configs'].items():
                if section not in ('train', 'document_refs'):
                    continue
                merged = copy.deepcopy(inventory)
                for original, shard in zip(inventory['shards'], merged['shards']):
                    src = checked_path(root, original['path'])
                    dest = target / original['path']
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dest)
                    shard['path'] = relative + '/' + original['path']
                combined['configs'][config if section == 'train' else config + '_document_refs'] = merged
            write_json(target / 'release_manifest.json', manifest)
            write_json(target / 'source_manifest.json', source)
            write_json(target / '_SUCCESS', {'format_version': RELEASE_FORMAT_VERSION})
        combined['negative_counts'] = sorted(set(combined['negative_counts']))
        write_json(staging / 'release_manifest.json', combined)
        write_json(staging / 'source_manifest.json', sources)
        for name in ('finepdf_release.py', 'release_data.py', 'materialize_release_training.py'):
            shutil.copyfile(Path(__file__).with_name(name), staging / name)
        guide = Path(__file__).resolve().parent.parent / 'docs/data-guide.md'
        text = guide.read_text().replace('[pipeline.md](pipeline.md)',
            '[pipeline instructions](https://github.com/WissamAntoun/MegaQuery/blob/main/docs/pipeline.md)')
        (staging / 'GUIDE.md').write_text(
            '> Counts below describe the published MegaQuery-680M dataset. For this newly assembled export, '
            'use its release manifests for counts and provenance.\n\n' + text)
        front = ['---', 'license: ' + json.dumps(data_license), 'language: [en, fr]', 'configs:']
        for config in CONFIGS:
            for suffix, section in (('', 'train'), ('_document_refs', 'document_refs')):
                front.extend([f'  - config_name: {config}{suffix}', '    data_files:',
                              '      - split: train', f'        path: data/{config}/{section}/*.parquet'])
        front.extend(['---', '', '# MegaQuery', '',
                      'English and French grouped retrieval queries and document references.', '',
                      'Indices are local to each language and address its pinned FinePDFs order.', '',
                      '[Data guide](GUIDE.md) · [Code](https://github.com/WissamAntoun/MegaQuery)', '',
                      'Counts and provenance for this export are in the language manifests.', ''])
        (staging / 'README.md').write_text('\n'.join(front))
        write_json(staging / '_SUCCESS', {'format_version': RELEASE_FORMAT_VERSION,
                                         'layout': 'multilingual', 'language_configs': list(CONFIGS)})
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--english', type=Path, required=True)
    parser.add_argument('--french', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    combine(args.english, args.french, args.output_dir)


if __name__ == '__main__':
    main()
