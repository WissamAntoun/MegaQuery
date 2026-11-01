# Understanding MegaQuery-680M

This guide describes the combined English and French release. Counts come from its
published manifests and schemas. The small examples are **fictional teaching
examples**, with fewer queries and negatives to keep them readable. They are
not copies of real training rows.

## What you have released

We release questions, links to documents, and scores describing how
those questions and documents relate. The document text stays in the pinned
FinePDFs dataset. A small helper follows the links and retrieves the text when
someone wants to train a model.

A **positive** is the passage used to generate a question. A **negative** is
another, similar-looking passage supplied as an alternative during training.
A **hard negative** is an alternative found automatically by similarity search.

| Item | English (`eng_Latn`) | French (`fra_Latn`) |
| --- | ---: | ---: |
| Training rows: query groups | 145,619,690 | 27,323,217 |
| Queries, including additional queries | 569,212,798 | 108,001,708 |
| Referenced source documents | 145,543,330 | 27,312,687 |
| Documents in the full pinned source split | 206,917,202 | 27,312,735 |
| Training Parquet files | 1,457 | 274 |
| Document-reference Parquet files | 1,456 | 274 |
| FinePDFs source Parquet files | 579 | 84 |
| Negatives per training row | 57–64 | 16–64 |

Together: **677,214,506 queries** in **172,942,907 training rows**.

The training configs are `eng_Latn` and `fra_Latn`. The corresponding document
address books are `eng_Latn_document_refs` and `fra_Latn_document_refs`.

**IDs are local to each language.** Use `(config, query_id)` to identify a query
across languages, and keep the config with document indices when mixing rows.
The same integer can refer to different English and French documents.

Repeated negative references and candidates resolving to the positive document
are removed.

One training row usually holds a main question plus three additional questions.
The actual number can vary. The training-row count therefore differs from the
total-query count. Multiple original mining groups can also point to the same
released positive document, so training rows are not a count of distinct documents.

The original source is `HuggingFaceFW/finepdfs`, configurations `eng_Latn` and `fra_Latn`, split
`train`, at commit:

```text
220bac3acbf07789502c621d2d33952f51ac7f86
```

We release this dataset under **odc-by**.

For the code repository's copies of the helpers, run from its root after setting
`PYTHONPATH=src`; the examples below otherwise assume the downloaded dataset
layout, where the helpers sit beside the manifests.

## Recommended training workflow

For training, use `load_source_dataset()` to prepare the complete source corpus
for this language once. It builds a locally cached Arrow dataset in exactly the
order specified by `source_manifest.json`. The corpus is memory-mapped from
disk, so it can be larger than RAM. Later runs reuse the cache.

From the downloaded release directory, install the reading dependencies with
`pip install datasets huggingface_hub pyarrow`, then:

```python
from finepdf_release import iter_release_rows, load_source_dataset

release_dir = "."
CONFIG = "eng_Latn"  # Use "fra_Latn" for French.
documents = load_source_dataset(release_dir, config=CONFIG).select_columns(["text"])
row = next(iter_release_rows(release_dir, config=CONFIG))

document_ids = [row["positive_idx"], *row["negative_idxs"][:8]]
texts = documents[document_ids]["text"]
print({"query": row["query"], "positive": texts[0], "negatives": texts[1:]})
```

The first preparation downloads the **entire pinned source language split**
and builds the Arrow cache. Allow time and disk space for both. To reuse an
existing source download, pass `local_source_root="/path/to/finepdfs"`; this root
must contain paths such as `data/eng_Latn/train/...`. Use
`cache_dir="/path/to/arrow-cache"` to choose where the prepared Arrow files live.

Each document index is already a row position in `documents`, so no separate
ID lookup dictionary is needed. Selecting the `text` column keeps those positions.
Keep every source row in its original order; shuffle or filter training examples
separately. For batches, retrieve each list of documents together:

```python
examples["documents"] = [
    documents[document_ids]["text"]
    for document_ids in examples["document_ids"]
]
```

For inspecting examples or preparing small subsets, `FinePDFResolver` offers
selective lookups without preparing the whole Arrow corpus. Its cache misses
read source Parquet row groups, so we recommend the prepared Arrow corpus for
repeated random lookups during large training runs.

## A map of the directory

```text
MegaQuery-680M/
├── README.md
├── GUIDE.md
├── release_manifest.json
├── source_manifest.json
├── _SUCCESS
├── finepdf_release.py
├── release_data.py
├── materialize_release_training.py
└── data/
    ├── eng_Latn/
    │   ├── train/*.parquet
    │   ├── document_refs/*.parquet
    │   ├── release_manifest.json
    │   ├── source_manifest.json
    │   └── _SUCCESS
    └── fra_Latn/
        └── same layout as eng_Latn/
```

| File or directory | Plain-language purpose |
| --- | --- |
| `data/<config>/train/` | Grouped queries, document indices, and aligned scores. |
| `data/<config>/document_refs/` | The address book for referenced source documents. |
| `data/<config>/source_manifest.json` | The exact FinePDFs version and ordered source files defining that language's document indices. |
| `data/<config>/release_manifest.json` | Original language counts, settings, file inventories, and checksums. Its file paths are relative to its language directory. |
| Root `release_manifest.json` | Combined counts, language-directory mapping, and inventories with repository-relative paths. |
| Root `source_manifest.json` | An index pointing to each language's source manifest; it does not combine document numbering. |
| Root `README.md` | Dataset introduction, Hugging Face configurations, and loading examples. |
| `GUIDE.md` | Schemas and readable examples, shared by both languages. |
| `finepdf_release.py` | Shared reading helpers. Pass `config="eng_Latn"` or `config="fra_Latn"` at the root. |
| `materialize_release_training.py` | A text-subset command; choose the language with `--config`. |
| `release_data.py` | Shared schemas and the writer needed by the helpers. |
| `_SUCCESS` | Completion receipt at the root and in each language directory. |

In the sections below, `train/`, `document_refs/`, and the detailed source and
release manifests refer to the selected directory `data/<config>/`.

A **Parquet file** is a compressed table, rather like a spreadsheet. It can also
store a list or a small record inside a cell. A **shard** is one piece of a large
table. Files such as `part-00000.parquet` and `part-00001.parquet` have the same
columns and belong to the same dataset. Most training/reference shards have
100,000 rows; the last shard holds the remainder.

In the schemas below:

- `int64` means a whole number, used for IDs, positions, or counts.
- `float64`/`double` means a decimal number, used for scores.
- `string` means text.
- `list[...]` means several values in one cell.
- `record`/`struct` means a small collection of named fields in one cell.
- `null` means a value was unavailable or not retained; it is not zero.

## First: understand the different IDs

| Name | What it identifies |
| --- | --- |
| `query_id` | A question's new ID within this language release. Primary and additional queries each have their own ID. |
| `mining_query_id` | That question's old numeric ID from the mining pipeline. Useful for tracing its history. |
| `source_query_id` | The original generation ID, when retained. It is text, often a document ID plus a `__s1d_...` suffix. In compacted inputs this was lost, so it is normally `null`. |
| `positive_idx`, `negative_idxs`, `skipped_idxs`, `finepdf_idx` | Positions in the pinned, ordered FinePDFs source corpus for this language/split. These can retrieve source text. |
| `finepdf_id` | The original document identifier string. It may look like a UUID or `<urn:uuid:...>`. Preserve its exact spelling. This label can repeat. |

Document positions count from **zero**. Their meaning comes from
`source_manifest.json`, which specifies the file order.

Here is a tiny fictional source corpus:

| Source file | First global document position | Number of rows |
| --- | ---: | ---: |
| `example-00000.parquet` | 0 | 100 |
| `example-00001.parquet` | 100 | 100 |

Document **120** is therefore row **20** of the second file: `100 + 20 = 120`.

**A document index is not its row number inside `document_refs`.** That address
book contains only referenced documents. Its first row could describe document
120. Use the helper or the recorded source coordinates to retrieve text.

Keeping the source version and file order matters: a reordered or filtered
corpus would give the same number a different meaning.

## `train/*.parquet`: the training examples

Each row contains a group of questions about one positive passage. `query` is
the group's representative question used for reranking. `other_queries` holds
the remaining questions in their stored order.

### Schema

| Column | Type | Meaning |
| --- | --- | --- |
| `query_id` | int64 | New ID of the primary question. |
| `mining_query_id` | int64 | Old mining ID of that question. |
| `query` | string | Primary question text. |
| `query_word_count` | int64 | Actual number of whitespace-separated words in this question. |
| `generation_metadata` | list[GenerationOrigin] | Retained generation records for the primary question; structure below. |
| `other_queries` | list[OtherQuery] | Other questions about this positive passage; structure below. |
| `positive_idx` | int64 | Address of the positive passage in the source corpus. |
| `negative_idxs` | list[int64] | Addresses of the retained negative passages. |
| `negative_retrieval_scores` | list[float64] | Similarity between the positive passage and each negative passage. |
| `negative_reranker_scores` | list[float64] | Reranker scores between the primary question and each negative passage. |
| `negative_reranker_raw_scores` | list[float64], or null | The same negative reranker scores before sigmoid conversion, when retained. |
| `positive_reranker_score` | float64 | Reranker score between the primary question and its positive passage. |
| `positive_reranker_raw_score` | float64, or null | Positive reranker score before sigmoid conversion, when retained. |
| `skipped_idxs` | list[int64] | Nearby candidates excluded before reranking. They are not training negatives. |
| `skipped_retrieval_scores` | list[float64] | Similarity scores for those excluded candidates. |

`GenerationOrigin` is a nested record with these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `source_query_id` | string, or null | Original generation ID, if retained. |
| `query_type` | string, or null | Requested kind of question, such as `factual`, `causal`, or `keywords`. |
| `difficulty` | string, or null | Requested difficulty label, such as `college` or `phd`. |
| `character` | string, or null | The imagined person/perspective used to generate the question. It can be written in English even for a question in another language. |
| `query_length_target` | int64, or null | Requested question length in words. This can differ from the actual word count. |
| `metadata_json` | string | The selected original metadata, stored as JSON text. Decode with `json.loads()` to inspect extra fields. |

There can be several origins because different generation requests produced the
same question. Their original records remain attached. Requested type,
difficulty, and length describe generation settings, rather than human ratings.

`OtherQuery` is a nested record with these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `query_id` | int64 | New ID of this additional question. |
| `mining_query_id` | int64 | Its old mining query ID. |
| `query` | string | Its question text. |
| `query_word_count` | int64 | Its actual word count. |
| `generation_metadata` | list[GenerationOrigin] | Its own generation records, in the same structure described above. |

### Small example

This fictional row uses only **two questions and two negatives**. The schema
allows more. The JSON strings inside `metadata_json` are shortened examples of
the extra original metadata.

```json
{
  "query_id": 0,
  "mining_query_id": 9000,
  "query": "Quand commencent les cours?",
  "query_word_count": 4,
  "generation_metadata": [
    {
      "source_query_id": null,
      "query_type": "factual",
      "difficulty": "college",
      "character": "Un étudiant qui prépare son année universitaire.",
      "query_length_target": 20,
      "metadata_json": "{\"source_query_id\":null,\"stage1_question_type\":\"factual\"}"
    }
  ],
  "other_queries": [
    {
      "query_id": 1,
      "mining_query_id": 9001,
      "query": "Quelles sont les dates des examens?",
      "query_word_count": 6,
      "generation_metadata": [
        {
          "source_query_id": null,
          "query_type": "factual",
          "difficulty": "phd",
          "character": "Un étudiant qui prépare son année universitaire.",
          "query_length_target": 20,
          "metadata_json": "{\"source_query_id\":null,\"stage1_difficulty\":\"phd\"}"
        }
      ]
    }
  ],
  "positive_idx": 120,
  "negative_idxs": [143, 187],
  "negative_retrieval_scores": [0.82, 0.74],
  "negative_reranker_scores": [0.20, 0.05],
  "negative_reranker_raw_scores": [-1.386294, -2.944439],
  "positive_reranker_score": 0.95,
  "positive_reranker_raw_score": 2.944439,
  "skipped_idxs": [121, 122],
  "skipped_retrieval_scores": [0.96, 0.94]
}
```

Read this as: “Use document 120 as the positive. Compare it with alternatives
143 and 187. Documents 121 and 122 were retrieved but excluded. The main
question and the additional question share these document references.”

### How to read the scores

All negative lists line up by position:

| List position | Document | Retrieval score | Reranker score | Raw reranker score |
| --- | ---: | ---: | ---: | ---: |
| 0 | 143 | 0.82 | 0.20 | -1.386294 |
| 1 | 187 | 0.74 | 0.05 | -2.944439 |

The **retrieval score** compares documents with documents. It explains why an
alternative was retrieved. The **reranker score** compares the main question
with a document. Higher values mean a stronger match according to that model.

The regular reranker score is the raw score passed through **sigmoid**, a
conversion to a value between 0 and 1. A score of `0.95` is a model score; it
has not been established as a calibrated “95% chance of relevance.”

Negatives are ordered by descending **retrieval score**, rather than reranker
score. Keep the score lists together when selecting or reordering negatives.

**The positive and negative reranker scores apply only to `query`.** The extra
questions can reuse the document references for ordinary retrieval training,
but they do not have their own reranker scores in this release.

We retrieve the Top-k = 96 non-self candidates, skip the first 32 ranks, and rerank the remaining 64 candidates. Retrieval uses the positive document's embedding; reranking uses the primary question. Removing repeated negative references can reduce that count.
The smallest observed negative count is 57 for English and 16 for French.

## `document_refs/*.parquet`: the document address book

Each row tells you where one referenced document lives. It does not contain
document text. The table is sorted by `finepdf_idx` and includes documents
needed for positives, retained negatives, and excluded-neighbour slots.

### Schema

| Column | Type | Meaning |
| --- | --- | --- |
| `finepdf_idx` | int64 | Public document position in the ordered source corpus. |
| `finepdf_id` | string | Original source identifier; it is not guaranteed unique. |
| `source_shard` | string | Source Parquet path, relative to the FinePDFs repository root. |
| `row_in_shard` | int64 | Zero-based row inside that source file. |

### Example

```json
{
  "finepdf_idx": 120,
  "finepdf_id": "<urn:uuid:11111111-1111-4111-8111-111111111111>",
  "source_shard": "data/fra_Latn/train/example-00001.parquet",
  "row_in_shard": 20
}
```

This is the same document 120 from the training example. In the fictional
two-file corpus above, the second file starts at 100, so row 20 is document 120.
The helper can calculate this address directly from `source_manifest.json`;
`document_refs` is also a convenient explicit lookup table.

## `source_manifest.json`: the source version and file order

This is the map used to interpret public document positions. JSON is an
ordinary readable text format. Its important fields are:

| Field | Meaning |
| --- | --- |
| `repo_id` | Original dataset repository. |
| `config` | Language/script selection, `fra_Latn` here. |
| `split` | Source split, `train` here. |
| `revision` | Exact source commit, protecting the meaning of the document positions. |
| `num_rows` | Size of the entire source split, including unreferenced rows. |
| `shards` | Ordered source-file records containing `path`, `start_row`, and `num_rows`. |

This fictional example matches the document-120 explanation:

```json
{
  "repo_id": "HuggingFaceFW/finepdfs",
  "config": "fra_Latn",
  "split": "train",
  "revision": "220bac3acbf07789502c621d2d33952f51ac7f86",
  "num_rows": 200,
  "shards": [
    {"path": "data/fra_Latn/train/example-00000.parquet", "start_row": 0, "num_rows": 100},
    {"path": "data/fra_Latn/train/example-00001.parquet", "start_row": 100, "num_rows": 100}
  ]
}
```

The French source manifest contains 84 files and 27,312,735 rows.
The English source manifest contains 579 files and 206,917,202 rows. The first French source file
is `data/fra_Latn/train/000_00000.parquet`, starting at 0 with 445,000 rows.

## `release_manifest.json`: the release inventory and history

Each language retains its original release manifest. The root manifest also
sums the counts, maps `language_configs` to directories, and prefixes shard
paths with `data/<config>/`. The per-language manifest fields are:

| Field | Meaning |
| --- | --- |
| `format_version` | Version of the release layout, 2 for this release. |
| `references_only` | `true`: document text is stored upstream, rather than repeated in this release. |
| `partial` | Whether this was deliberately or actually an incomplete mining export; `false` here. |
| `missing_mining_shards` | Missing mining shard IDs; `[]` here. |
| `query_count` | Primary plus additional queries: 569,212,798 for English or 108,001,708 for French. |
| `negative_counts` | The distinct counts of negatives observed in rows, not a histogram. |
| `source_manifest` | Filename of the source map. |
| `configs` | Inventories for `train` and `document_refs`, including row counts and shard lists. |
| `provenance` | License, models, generation/mining/reranking settings, and code/prompt information. |
| `data_processing` | Document numbering, negative cleanup, and negative score ordering. |

Each output shard's inventory entry contains a path, its first row in that
output table (`start_row`), its row count, and a file checksum (`sha256`).
**That output-table `start_row` is different from the source-corpus positions
inside its records.**

A file checksum lets you check whether the Parquet bytes changed after export.

We use these models:

- Generation model: `google/gemma-3-1b-it`.
- Embedding model: `Qwen/Qwen3-Embedding-0.6B`.
- Reranker: `wissamantoun/bge-reranker-v2-gemma-seq-cls`.

Check `provenance.configuration_source` and its notes to distinguish verified
historical settings from repository defaults. A `null` model or code revision
means that revision was not recorded; it should not be inferred from filenames.

Long source documents can be shortened before a model reads them. The exact
generation, embedding, and reranker limits are recorded in `provenance`. The
document references still retrieve the full source text; the stored scores
remain those computed using the configured shortened views.

## `README.md` and `_SUCCESS`

`README.md` is the short dataset description shipped with the release. Its
opening block defines the training configurations `eng_Latn` and `fra_Latn`,
and their `eng_Latn_document_refs` and `fra_Latn_document_refs` address books. This longer guide explains the data columns
and supplied helpers in more detail.

`_SUCCESS` is a tiny JSON file:

```json
{"format_version": 2}
```

This is the per-language completion receipt. The root receipt also identifies
the multilingual layout and its two language configurations. It marks a completed release. The reader checks for it before loading training
rows. The file inventories and checksums provide the information needed for
additional integrity checks after copying the release.

## The three Python files

### `finepdf_release.py`

This is the main reading helper. It offers:

- `load_source_dataset(...)`: recommended for training; prepare the complete source corpus as a cached Arrow dataset in the manifest's order, with direct positional text lookup.
- `iter_release_rows(...)`: read training rows in the manifest's order.
- `FinePDFResolver(...)`: selectively retrieve source text, or original `id` and `text` records, for inspection and small subsets.
- `iter_training_examples(...)`: turn grouped reference rows into ordinary text examples, optionally expanding additional questions.

`FinePDFResolver` uses `source_manifest.json` directly; it does not need to scan
the entire address book for each lookup. It caches a bounded number of source
files and row groups. Requesting scattered negatives can still require many
source files, even for a small training sample.

### `materialize_release_training.py`

This command uses `FinePDFResolver` to write a new text-containing subset.
For large training runs, use the prepared Arrow corpus described above.
An output row has:

```json
{
  "query": "Quand commencent les cours?",
  "positive": "Le texte du calendrier universitaire...",
  "negative_0": "Le texte d'un autre document...",
  "negative_1": "Le texte d'un deuxième document..."
}
```

All these output columns are strings. The materialized view contains question
and passage text; it omits teacher scores, document IDs, and generation metadata.
The original reference release remains available for those fields.

`--all-queries` also emits the additional questions as separate examples, using
the same passage references. `--limit` limits emitted examples. Requesting more
negatives than a row contains raises an error; the helper does not pad or repeat
passages. The example below requests 8 negatives, which fits this
release's minimum of 16. If this minimum is zero, the example
produces query/positive pairs only; select rows with negatives for contrastive training.

### `release_data.py`

This contains the column definitions and the code for writing compressed output
shards, counting rows, and calculating checksums. It supports the reader and materializer.
Keep it beside `finepdf_release.py` and `materialize_release_training.py` when copying those helpers.
There is normally no reason to run it directly.

## A small text-subset example

From the downloaded release directory, this makes a separate text dataset
containing up to 1,000 examples. Replace the two `/path/to/...` placeholders
with your own local directories. The source root must contain the source
manifest's relative paths, such as `data/eng_Latn/train/...`.

```bash
python materialize_release_training.py \
  --release-dir . \
  --config eng_Latn \
  --local-source-root /path/to/finepdfs \
  --output-dir /path/to/training-sample \
  --num-negatives 8 \
  --limit 1000
```

Choose a new output directory if that one already exists. Materializing many
examples can take substantial time and disk space. If no local source root is supplied,
the resolver downloads required Parquet files from the pinned source commit;
the number of examples does not cap download size.

For ordinary retrieval training, start with `train/`, the manifests, and
`load_source_dataset()` from the reading helper. Detailed generation and mining instructions will be available
in [pipeline.md](pipeline.md).
