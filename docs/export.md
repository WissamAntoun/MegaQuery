# Export and the combined English/French layout

This guide adapts the release README. Export runs on CPU, without Ray, vLLM or
GPU inference. The exported tables reference full FinePDFs text upstream.

## Required artifacts

A language export needs:

1. The matching `index/doc_id_map.parquet` and all canonical embedding Parquets.
2. Completed v4 mining shards, including raw scores if these are to be retained.
3. Raw or compacted generation Parquets containing each mined query and its
   retained metadata. Metadata must match exactly by source ID and query text.
4. The entire pinned FinePDFs language split, including unreferenced rows, or a
   compatible prebuilt ID/shard/row source index.
5. Provenance JSON describing the new run's models, settings, score semantics,
   known revisions and data license. The example is a starting point; it is not
   proof of a completed historical run.

`bash examples/run_stage.sh configs/english.env export` invokes the exporter
with these inputs, checks the source against the published manifest, and writes
a new `work/eng_Latn/release/` directory. Existing output directories are rejected.
`--rows-per-shard` defaults to 100,000. Memory limits, spill parent, thread count,
generation partitions and training conversion workers are CLI options.

## Source remapping and cleanup

The source is concatenated in the explicit pinned shard order. The exporter
maps mining integers through source IDs to that order. For repeated source IDs,
it compares exact retained text hashes; distinct text variants get their own
source positions, and identical matches choose the smallest source position.
If a repeated ID cannot be disambiguated, export fails. It does not deduplicate
or filter the source corpus. Unique-ID joins rely on the supplied source snapshot.

The export removes self-negatives and repeated negative references **after
source remapping**, retains one candidate per source document, and aligns
retrieval/reranker/raw-score lists by document identity. Negatives remain in
descending retrieval-score order. Skipped-neighbor slots remain excluded slots;
remapping can make their indices repeat or overlap other references.
No additional reranker-score threshold is applied.

Default validation rejects missing documents, incomplete mining, missing query
metadata, and incompatible query ordering. Explicit compatibility policies are
available for already generated artifacts: `--missing-document-policy
 drop-empty-negatives` only removes verified empty queryless slots, and
`--serialized-query-policy exclude-groups` excludes malformed serialized-query
groups. These are opt-in policies; do not enable them as a general missing-data
workaround. Newly generated clean data should use the defaults.

Compacted generation inputs have lost original variant suffix IDs and selected
fields. The exporter leaves those values null instead of inventing provenance.
Multiple generation origins for a deduplicated query are retained.

## Optional source ID index

For moving source coordinates without full text:

```bash
python src/build_finepdf_source_index.py \
  --finepdf-root work/finepdfs \
  --finepdf-glob 'work/finepdfs/data/eng_Latn/train/*.parquet' \
  --source-config eng_Latn \
  --source-revision 220bac3acbf07789502c621d2d33952f51ac7f86 \
  --output-dir work/eng_Latn/source-index
```

For export, replace `--finepdf-root`/`--finepdf-glob` with `--source-index-dir`.
The source ID index uses a sorted complete file list; the standard export
example's `--expected-source-manifest` enforces published-order agreement.
Repeated IDs needing text disambiguation require the original raw source text;
an ID-only index cannot resolve those cases. The full pinned source is therefore
the recommended input for the completed English/French recipe.

## Combine locally

```bash
python src/combine_releases.py --english work/eng_Latn/release \
  --french work/fra_Latn/release --output-dir work/MegaQuery-680M
```

The assembler accepts complete format 2 exports, checks file checksums and row
inventories, and copies only the two data inventories plus public metadata and
readers. It writes into a new directory and leaves inputs untouched. Copying is
intentional; reserve another output copy's disk space.

```text
MegaQuery-680M/
  README.md
  GUIDE.md
  release_manifest.json
  source_manifest.json
  _SUCCESS
  finepdf_release.py
  release_data.py
  materialize_release_training.py
  data/
    eng_Latn/
      train/*.parquet
      document_refs/*.parquet
      release_manifest.json
      source_manifest.json
      _SUCCESS
    fra_Latn/
      ...same structure...
```

The root card defines `eng_Latn`, `fra_Latn`, `eng_Latn_document_refs`, and
`fra_Latn_document_refs`. The root release manifest maps each language to its
directory and prefixes its inventory paths; the root source manifest points to
the per-language source manifests. Language manifests retain relative paths and
their own provenance. The assembler performs no ID offsetting, interleaving or
reordering. Use `(config, query_id)` or `(config, document_idx)` across languages.

Both `FinePDFResolver` and `load_source_dataset` require `config="eng_Latn"` or
`config="fra_Latn"` at the combined root. They also accept a standalone language
directory. The root requires `_SUCCESS`; each selected language must be complete.
See [the data guide](data-guide.md) for examples and all schemas.

The assembler is new release-preparation code implementing the documented
combined layout; the original layout assembly command was not present in the
available source. It is tested with fixtures, not claimed as the historical tool.
