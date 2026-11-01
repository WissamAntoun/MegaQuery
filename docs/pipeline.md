# Run the pipeline in order

These instructions adapt the generation, embedding/mining and release READMEs
to one portable path. Commands run from the repository root. First complete
[environment setup](environment.md), and review [reproduction limits](reproduction.md).

## 1. Obtain the published source manifests

The published release pins FinePDFs commit
`220bac3acbf07789502c621d2d33952f51ac7f86`. Obtain its small manifest files first:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "wissamantoun/MegaQuery-680M", repo_type="dataset",
    local_dir="work/published-release",
    allow_patterns=["release_manifest.json", "source_manifest.json", "_SUCCESS",
                    "data/*/release_manifest.json", "data/*/source_manifest.json",
                    "data/*/_SUCCESS"],
)
```

For a stable new run, also record or pin the MegaQuery repository commit used
for these manifests. The source manifest lists each source shard's path,
`start_row`, `num_rows`, repository, language and immutable commit. English
contains 579 source shards / 206,917,202 rows; French contains 84 / 27,312,735.
The training release references a subset of these documents.

```bash
cp configs/english.env.example configs/english.env
# Edit paths and resource settings in configs/english.env.
bash examples/run_stage.sh configs/english.env prepare
```

This downloads the manifest's exact FinePDFs files and checks row counts and
`id`/`text` columns. Existing local snapshots can instead be validated with:

```bash
python src/prepare_source.py --release-dir work/published-release \
  --config eng_Latn --output-dir work/finepdfs --local-only
```

Local-only validation trusts that you supplied the declared snapshot; row counts
do not prove byte identity. Keep the source rows and shard order intact. The
export example checks the complete ordered source inventory against the
published manifest before assigning public indices. Generation and the mining
index have separate file/row orders and do not define public document indices.

## 2. Supply personas and compute their vectors

Supply `work/personas.jsonl`, one nonempty `{"persona": "profile text"}` object
per line, in a stable order. The exact historical persona file and its revision
are not distributed here. A replacement persona collection changes generation;
it is not an exact reproduction. Review attribution for whatever collection you use.

```bash
bash examples/run_stage.sh configs/english.env personas
```

Output: `work/persona_embeddings.npy`. Compute this once and reuse it for both
languages, provided the persona JSONL, embedding model and vector order match.
The matrix must have one row per persona and the same dimension as passage
embeddings. [Generation details](generation.md) include the persona instruction.

## 3. Generate four requested variants per passage

In the GPU generation environment:

```bash
bash examples/run_stage.sh configs/english.env generate
```

Input: FinePDFs Parquets with `id`, `text`. Output:
`work/eng_Latn/queries/<rank>/*.parquet`, including source content, query text and
generation metadata. [Generation](generation.md) describes sampling and prompts.

For multiple independent workers, use a separate config per worker with a unique
`GEN_RANK` in `0..GEN_WORLD_SIZE-1`; run every rank with the same world size.
Each receives sorted input files `[rank::world_size]`. Do not use more generation
ranks than source files. Wait for all generation workers before compaction.
Choose fresh output paths for a new run; generation does not promise exactly-once
recovery from arbitrary partial failures.

## 4. Compact and embed

```bash
bash examples/run_stage.sh configs/english.env compact
bash examples/run_stage.sh configs/english.env embed
```

Compaction is CPU only; embedding needs the GPU environment. Compaction writes
`work/eng_Latn/compact/compact_w*_b*.parquet`, grouping within a worker batch by
base document ID and storing query metadata in aligned lists. It does not merge
all documents globally. It drops generation suffix IDs and selected raw fields;
keep raw generation output if you need those records.

Embedding writes `index/embeddings/shard_<rank>.parquet` and its manifest. With
multiple embedding workers, set `EMBED_WORLD_SIZE` consistently and run every
`EMBED_RANK` once. **Freeze that world size and the input file set** after shards
have been generated. Every document's first-seen content and ordered distinct
queries are retained within its assigned worker.

## 5. Build the index, retrieve and rerank

After every embedding worker has finished:

```bash
bash examples/run_stage.sh configs/english.env index
bash examples/run_stage.sh configs/english.env mine
```

The CPU index builder verifies shard coverage before writing `index.usearch`,
`doc_id_map.parquet`, `doc_texts/*.parquet`, and `doc_texts/meta.json`.
The GPU mining stage writes `mining/shard_*/` Parquets, checkpoints, and temporary
retrieval intermediates. Optional chunked reranking is described in
[mining.md](mining.md). Independent mining workers use their own `MINE_RANK` and
shared `MINE_WORLD_SIZE`; the total may differ from the embedding world size.
Wait for all mining shards before export.

## 6. Export each language and assemble the combined dataset

Review `configs/provenance.example.json` and copy it for your run. Set the
`PROVENANCE` path in your config and record your actual settings and revisions.
Then, in the CPU environment:

```bash
bash examples/run_stage.sh configs/english.env export
```

Output: a complete single-language reference release at
`work/eng_Latn/release/`. Full source text is kept upstream.

Repeat preparation, generation, compaction, embedding, indexing, mining and
export with `configs/french.env.example`, copied to `configs/french.env`.
The prepared persona matrix can be reused. All other output roots are separate.

```bash
python src/combine_releases.py \
  --english work/eng_Latn/release --french work/fra_Latn/release \
  --output-dir work/MegaQuery-680M
```

The assembler copies inventory-listed Parquets, verifies checksums, and creates
the four dataset configurations. It does not change input releases or document
indices. It never uploads anything. See [export.md](export.md) for the exact
layout and [data-guide.md](data-guide.md) for training and source-document loading.
