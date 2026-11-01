"""Small, local fixtures for transformations; no model weights or GPU required."""
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from usearch.index import Index

from combine_releases import combine
from compact_synth_parquets import _process_files, _flush_batch
from export_finepdf_release import parse_args, run, training_row
from finepdf_release import FinePDFResolver, iter_release_rows, iter_training_examples, load_source_dataset
from generate_hard_negatives_v4 import phase1_retrieve, make_score_postprocess
from ingest_parquet_embed_v3_shard import aggregate_docs
from prepare_source import prepare
from release_data import TRAIN_SCHEMA, sha256
from synth_gen_qwen3 import PersonaRetrieve, expand_stage1_disabled_rows, make_s2_preprocess

ROOT = Path(__file__).resolve().parents[1]


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)


def make_language(root, config, compact=False, workers=1, duplicate_source_id=False):
    source, index, mining = root / 'source', root / 'index', root / 'mining'
    for filename, names in [('000.parquet', ['A', 'C']), ('001.parquet', ['B', 'D'])]:
        records = [{'id': name, 'text': config + ' text ' + name} for name in names]
        if duplicate_source_id and filename == '000.parquet':
            records.append({'id': 'A', 'text': 'A different retained extraction'})
        write_parquet(source / 'data' / config / 'train' / filename, records)
    docs = ['B', 'C', 'A', 'D']
    write_parquet(index / 'doc_id_map.parquet',
                  [{'doc_id_int': i, 'doc_id_str': d} for i, d in enumerate(docs)])
    queries = {'A': ['A0', 'A1'], 'B': ['B0', 'B1'], 'C': [], 'D': []}
    write_parquet(index / 'embeddings/shard_00000.parquet', [
        {'doc_id_str': d, 'all_query_texts': queries[d], 'pos_text': config + ' text ' + d,
         'repr_query_text': queries[d][0] if queries[d] else ''} for d in docs])
    write_parquet(mining / 'shard_00000/hard_negatives.parquet', [
        {'doc_id_int': i, 'query_texts': queries[d], 'qids': [2, 3] if d == 'B' else [0, 1],
         'pos_reranker_score': .9, 'pos_reranker_raw_score': 2.2,
         'neg_embed': [{'doc_id': 1, 'score': .8}, {'doc_id': 3, 'score': .7}],
         'neg_reranker': [{'doc_id': 3, 'score': .1}, {'doc_id': 1, 'score': .2}],
         'neg_reranker_raw': [{'doc_id': 3, 'score': -2.2}, {'doc_id': 1, 'score': -1.4}],
         'skipped_doc_ids': [{'doc_id': 2 if i == 0 else 0, 'score': .95}]}
        for i, d in enumerate(docs) if queries[d]])
    (mining / 'shard_00000/_MATERIALIZED_SUCCESS').write_text('{}')
    generation = root / 'generation.parquet'
    if compact:
        rows = [{'id': d, 'query': queries[d], 'stage1_question_type': ['factual', 'summary'],
                 'stage1_difficulty': ['college', 'phd'], 'query_length_target': [20, 30]}
                for d in ('A', 'B')]
    else:
        rows = [{'id': d + '__s1d_' + str(i), 'query': q, 'stage1_question_type': 'factual',
                 'stage1_difficulty': 'college', 'query_length_target': 20}
                for d in ('A', 'B') for i, q in enumerate(queries[d])]
    write_parquet(generation, rows)
    provenance = root / 'provenance.json'
    provenance.write_text(json.dumps({'license': 'odc-by', 'generation': {'model': 'fixture-generator'},
        'mining': {'model': 'fixture-embedder', 'score_semantics': 'document cosine'},
        'reranker': {'model': 'fixture-reranker', 'score_semantics': 'sigmoid raw score'}}))
    output = root / 'release'
    args = parse_args(['--index-dir', str(index), '--hard-negatives-dir', str(mining),
                      '--queries-glob', str(generation), '--finepdf-root', str(source),
                      '--finepdf-glob', str(source / 'data' / config / 'train/*.parquet'),
                      '--source-config', config, '--source-revision', 'a' * 40,
                      '--provenance-json', str(provenance), '--output-dir', str(output),
                      '--rows-per-shard', '1', '--batch-size', '2', '--threads', '1',
                      '--train-workers', str(workers)])
    run(args)
    return output, source, args


@pytest.mark.parametrize('compact,workers', [(False, 1), (True, 2)])
def test_export_remapping_and_readers(tmp_path, compact, workers):
    output, source, args = make_language(tmp_path, 'eng_Latn', compact, workers)
    rows = list(iter_release_rows(output))
    assert [r['positive_idx'] for r in rows] == [0, 2]
    assert [r['query'] for r in rows] == ['A0', 'B0']
    assert rows[0]['negative_idxs'] == [1, 3]
    assert rows[0]['negative_retrieval_scores'] == [.8, .7]
    assert rows[0]['negative_reranker_scores'] == [.2, .1]
    assert rows[0]['negative_reranker_raw_scores'] == [-1.4, -2.2]
    assert rows[0]['skipped_idxs'] == [2]
    assert rows[0]['other_queries'][0]['query'] == 'A1'
    origin = rows[0]['generation_metadata'][0]
    assert origin['source_query_id'] == (None if compact else 'A__s1d_0')
    assert origin['query_type'] == 'factual'
    assert [q['query_id'] for r in rows for q in [r, *r['other_queries']]] == list(range(4))
    manifest = json.loads((output / 'release_manifest.json').read_text())
    assert manifest['query_count'] == 4
    for inventory in manifest['configs'].values():
        for shard in inventory['shards']:
            assert sha256(output / shard['path']) == shard['sha256']
    assert set(pq.read_schema(output / 'train/part-00000.parquet').names) == set(TRAIN_SCHEMA.names)
    assert '{{' not in (output / 'README.md').read_text()
    assert str(tmp_path) not in (output / 'release_manifest.json').read_text()
    with FinePDFResolver(output, source, max_cached_row_groups=1, max_open_shards=1) as resolver:
        assert resolver[[3, 0, 3]] == ['eng_Latn text D', 'eng_Latn text A', 'eng_Latn text D']
        examples = list(iter_training_examples(rows, resolver, 2, all_queries=True))
        assert len(examples) == 4 and examples[0]['positive'] == 'eng_Latn text A'
        with pytest.raises(ValueError):
            list(iter_training_examples(rows, resolver, 3))
        with pytest.raises(IndexError):
            resolver[-1]
    # A missing source file fails before accidental index reinterpretation.
    assert prepare(output, 'eng_Latn', source, local_only=True)['num_rows'] == 4
    expected = json.loads((output / 'source_manifest.json').read_text())
    expected['shards'][0]['num_rows'] += 1
    path = tmp_path / 'wrong-order.json'
    path.write_text(json.dumps(expected))
    args.output_dir = tmp_path / 'another-release'
    args.expected_source_manifest = path
    with pytest.raises(ValueError, match='Source files/order differ'):
        run(args)


def test_combined_language_local_indices(tmp_path):
    english, source_en, _ = make_language(tmp_path / 'en', 'eng_Latn')
    french, source_fr, _ = make_language(tmp_path / 'fr', 'fra_Latn', compact=True)
    combined = tmp_path / 'combined'
    manifest = combine(english, french, combined)
    assert manifest['query_count'] == 8
    assert set(manifest['configs']) == {'eng_Latn', 'fra_Latn', 'eng_Latn_document_refs', 'fra_Latn_document_refs'}
    with pytest.raises(ValueError, match='Choose a language'):
        next(iter_release_rows(combined))
    for config, source in [('eng_Latn', source_en), ('fra_Latn', source_fr)]:
        rows = list(iter_release_rows(combined, config=config))
        assert rows[0]['query_id'] == 0 and rows[0]['positive_idx'] == 0
        with FinePDFResolver(combined, config=config, local_source_root=source) as resolver:
            assert resolver[0] == config + ' text A'
        docs = load_source_dataset(combined, config=config, local_source_root=source,
                                   cache_dir=str(tmp_path / 'arrow-cache'))
        assert docs['text'] == [config + ' text ' + d for d in ['A', 'C', 'B', 'D']]
        text_output = tmp_path / ('materialized-' + config)
        subprocess.run([sys.executable, str(ROOT / 'src/materialize_release_training.py'),
            '--release-dir', str(combined), '--config', config,
            '--local-source-root', str(source), '--output-dir', str(text_output),
            '--num-negatives', '2', '--limit', '1'], check=True, capture_output=True)
        text_row = pq.read_table(text_output / 'train/part-00000.parquet').to_pylist()[0]
        assert text_row['positive'] == config + ' text A'
        assert text_row['negative_0'] == config + ' text C'
    with pytest.raises(ValueError):
        combine(english, french, combined)
    # Tampering with an input shard must prevent assembly.
    with (english / 'train/part-00000.parquet').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        combine(english, french, tmp_path / 'bad-combined')


def test_manifest_order_overrides_filename_order(tmp_path):
    source = tmp_path / 'source'
    release = tmp_path / 'release'
    release.mkdir()
    shards = []
    for offset, name in enumerate(('z.parquet', 'a.parquet')):
        relative = 'data/fra_Latn/train/' + name
        write_parquet(source / relative, [{'id': name, 'text': name}])
        shards.append({'path': relative, 'start_row': offset, 'num_rows': 1})
    (release / 'source_manifest.json').write_text(json.dumps({'config': 'fra_Latn',
        'repo_id': 'fixture', 'split': 'train', 'revision': 'a' * 40, 'num_rows': 2, 'shards': shards}))
    docs = load_source_dataset(release, local_source_root=source, cache_dir=str(tmp_path / 'arrow'))
    assert docs['text'] == ['z.parquet', 'a.parquet']
    with FinePDFResolver(release, source) as resolver:
        assert resolver[[1, 0]] == ['a.parquet', 'z.parquet']
    with pytest.raises(ValueError, match='streaming'):
        load_source_dataset(release, streaming=True)


def test_persona_sampling_prompt_and_compaction(tmp_path):
    personas = tmp_path / 'personas.jsonl'
    personas.write_text('\n'.join(json.dumps({'persona': p}) for p in ('Engineer', 'Teacher')))
    embeddings = tmp_path / 'personas.npy'
    np.save(embeddings, np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    retrieve = PersonaRetrieve(str(personas), str(embeddings), top_k=1)
    batch = retrieve({'__passage_embedding__': [[.01, 2.]], 'id': ['doc']})
    assert json.loads(batch['persona_candidates'][0]) == ['Teacher']
    variants = expand_stage1_disabled_rows({'__id__': 'doc', '__passage__': 'Passage',
        'text': 'Passage', 'persona_candidates': batch['persona_candidates'][0]})
    assert len(variants) == 4
    assert len({v['stage1_question_type'] for v in variants}) == 2
    assert len({v['stage1_difficulty'] for v in variants}) == 2
    assert all(v['stage1_character'] == 'Teacher' for v in variants)
    prompt = make_s2_preprocess('text', 'French', 'French', 'fixture-model')(variants[0])
    assert prompt['query_length_target'] in (10, 20, 30, 50)
    assert 'French' in prompt['request_kwargs']['messages'][1]['content']
    rows = [{'id': 'doc__s1d_0', 'text': 'Passage', 'query': 'query one', 'stage1_difficulty': None},
            {'id': 'doc__s1d_1', 'text': 'Passage', 'query': 'query two', 'stage1_difficulty': 'college'}]
    path = tmp_path / 'raw.parquet'
    write_parquet(path, rows)
    fields = ['query', 'stage1_difficulty']
    records, repeated = _process_files([str(path)], 'id', fields, [], '__s1d_', 2)
    output = tmp_path / 'compact.parquet'
    _flush_batch(records, repeated, fields, 'id', str(output))
    row = pq.read_table(output).to_pylist()[0]
    assert row['query'] == ['query one', 'query two']
    assert row['stage1_difficulty'] == [None, 'college']
    docs = aggregate_docs([str(output)], ['text'], 'query', 2, 'id')
    assert docs[0]['all_query_texts'] == ['query one', 'query two']
    assert docs[0]['repr_query_text'] == 'query one'


def test_actual_index_builder_and_retrieval(tmp_path):
    original = tmp_path / 'input.parquet'
    write_parquet(original, [{'id': 'A', 'text': 'A', 'query': 'A?'}])
    index_dir = tmp_path / 'index'
    vectors = np.array([[1., 0.], [.9, .1], [.8, .2], [0., 1.]], dtype=np.float32)
    rows = [{'doc_id_local': i, 'doc_id_str': str(i), 'pos_text': 'text ' + str(i),
             'pos_title': '', 'all_query_texts': [str(i) + '?'], 'repr_query_text': str(i) + '?',
             'embedding': vector.tolist()} for i, vector in enumerate(vectors)]
    write_parquet(index_dir / 'embeddings/shard_00000.parquet', rows)
    (index_dir / 'embeddings/shard_00000.manifest.json').write_text(json.dumps({
        'rank': 0, 'input_files': [str(original.resolve())], 'num_docs': 4,
        'num_queries': 4, 'embedding_dim': 2}))
    subprocess.run([sys.executable, str(ROOT / 'src/ingest_parquet_embed_v3_index.py'),
        '--glob', str(original), '--output-dir', str(index_dir), '--world-size', '1',
        '--doc-texts-shard-size', '2', '--index-batch-size', '1'], check=True, capture_output=True)
    index = Index.restore(str(index_dir / 'index.usearch'))
    neighbors, skipped = phase1_retrieve(vectors[:1], [0], index, 2, 4, 1, skip_top_n=1)
    assert [d for d, _ in skipped[0]] == [1]
    assert [d for d, _ in neighbors[0]] == [2, 3]
    assert 1 - neighbors[0][0][1] > 1 - neighbors[0][1][1]
    assert pq.read_table(index_dir / 'doc_id_map.parquet')['doc_id_int'].to_pylist() == [0, 1, 2, 3]


def test_score_activation():
    postprocess = make_score_postprocess(store_raw_reranker_scores=True)
    row = {'doc_id_int': 0, 'all_qids': '[0]', 'all_query_texts': '["query"]',
           'skipped_doc_ids': '[]', 'pair_type': 'neg', 'neg_doc_int': 1,
           'neg_embed_score': .8, 'data': [{'score': -2.} ]}
    result = postprocess(row)
    assert result['reranker_raw_score'] == -2.
    assert result['reranker_score'] == pytest.approx(1 / (1 + np.exp(2)))
    row['data'] = [{'score': 1000.}]
    assert postprocess(row)['reranker_score'] == 1.


def test_negative_alias_cleanup():
    record = {'doc_id_int': 0, 'query_texts': ['query'], 'qids': [10],
              'neg_embed': [{'doc_id': i, 'score': score} for i, score in [(1, .8), (2, .9), (3, .95)]],
              'neg_reranker': [{'doc_id': i, 'score': .1 * i} for i in (3, 2, 1)],
              'skipped_doc_ids': [], 'pos_reranker_score': .9}
    queries = [{'query': 'query', 'mining_query_id': 10, 'generation_metadata': '[]'}]
    # Slots 1 and 2 alias one source document; slot 3 aliases the positive.
    result = training_row(record, queries, np.array([0, 1, 1, 0]), 0)
    assert result['negative_idxs'] == [1]
    assert result['negative_retrieval_scores'] == [.9]
    assert result['negative_reranker_scores'] == [.2]
    with pytest.raises(ValueError, match='self/duplicate'):
        training_row(record, queries, np.array([0, 1, 1, 0]), 0, alias_policy='error')


def test_imports_and_cli_help():
    for path in sorted((ROOT / 'src').glob('*.py')):
        importlib.import_module(path.stem)
        if "if __name__ == " in path.read_text():
            result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True)
            assert result.returncode == 0, (path.name, result.stderr)
            assert 'usage:' in result.stdout


@pytest.mark.parametrize('config_name', ['english', 'french'])
def test_launcher_configuration_and_flags(tmp_path, config_name):
    capture = tmp_path / 'command.json'
    stub = tmp_path / 'capture-python'
    stub.write_text('#!' + sys.executable + '\nimport os,sys,json\n'
                    'open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    stub.chmod(0o755)
    config = tmp_path / 'settings.env'
    config.write_text((ROOT / f'configs/{config_name}.env.example').read_text()
                      + '\nPYTHON=' + str(stub) + '\nRAY_OBJECT_STORE_MEMORY_GB=8\n')
    environment = {**os.environ, 'CAPTURE': str(capture)}
    subprocess.run(['bash', '-n', str(ROOT / 'examples/run_stage.sh')], check=True)
    for stage in ('prepare', 'personas', 'generate', 'compact', 'embed', 'index', 'mine', 'export'):
        subprocess.run(['bash', 'examples/run_stage.sh', str(config), stage], cwd=ROOT,
                       env=environment, check=True, capture_output=True)
        command = json.loads(capture.read_text())
        tree = ast.parse((ROOT / command[0]).read_text())
        allowed, required = {'--help'}, set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                flags = {arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                         and arg.value.startswith('--')}
                allowed.update(flags)
                if any(k.arg == 'required' and isinstance(k.value, ast.Constant) and k.value.value for k in node.keywords):
                    required.update(flags)
        passed = {word for word in command[1:] if word.startswith('--')}
        assert passed <= allowed, (stage, passed - allowed)
        assert required <= passed, (stage, required - passed)


def test_duplicate_source_id_exact_text_resolution(tmp_path):
    output, source, _ = make_language(tmp_path, 'fra_Latn', duplicate_source_id=True)
    rows = list(iter_release_rows(output))
    # Both upstream variants keep their positions; only the exact text is referenced.
    assert [r['positive_idx'] for r in rows] == [0, 3]
    refs = [r for shard in sorted((output / 'document_refs').glob('*.parquet'))
            for r in pq.read_table(shard).to_pylist()]
    assert [r['finepdf_idx'] for r in refs] == [0, 1, 3, 4]
    with FinePDFResolver(output, source) as resolver:
        assert resolver[2] == 'A different retained extraction'
        assert resolver[3] == 'fra_Latn text B'
