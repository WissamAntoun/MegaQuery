# Understanding the {{language}} MegaQuery-680M release

This guide describes the files shipped in this release. Counts come from its
manifests and schemas. The small examples are **fictional teaching
examples**, with fewer queries and negatives to keep them readable. They are
not copies of real training rows.

## What you have released

You have released questions, links to documents, and scores describing how
those questions and documents relate. The document text stays in the pinned
FinePDFs dataset. A small helper follows the links and retrieves the text when
someone wants to train a model.

A **positive** is the passage used to generate a question. A **negative** is
another, similar-looking passage supplied as an alternative during training.
A **hard negative** is an alternative found automatically by similarity search.

| Item | This release |
| --- | ---: |
| Training rows: groups of queries sharing a positive document | {{train_rows}} |
| Total queries, counting primary and additional queries | {{query_count}} |
| Referenced source documents | {{ref_rows}} |
| Documents in the entire pinned source split | {{source_rows}} |
| Training Parquet files | {{train_files}} |
| Document-reference Parquet files | {{ref_files}} |
| Original source Parquet files | {{source_files}} |
| Observed numbers of negatives per training row | {{negative_counts}} |
| Incomplete/partial release? | {{partial}} |

Repeated negative references and candidates resolving to the positive document
are removed.

One training row usually holds a main question plus three additional questions.
The actual number can vary. The training-row count therefore differs from the
total-query count. Multiple original mining groups can also point to the same
released positive document, so training rows are not a count of distinct documents.

The original source is `{{repo}}`, configuration `{{config}}`, split
`{{split}}`, at commit:

```text
{{revision}}
```

We release this dataset under **{{license}}**.

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
documents = load_source_dataset(release_dir).select_columns(["text"])
row = next(iter_release_rows(release_dir))

document_ids = [row["positive_idx"], *row["negative_idxs"][:8]]
texts = documents[document_ids]["text"]
print({"query": row["query"], "positive": texts[0], "negatives": texts[1:]})
```

The first preparation downloads the **entire pinned source language split**
and builds the Arrow cache. Allow time and disk space for both. To reuse an
existing source download, pass `local_source_root="/path/to/finepdfs"`; this root
must contain paths such as `data/{{config}}/{{split}}/...`. Use
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

| File or directory | Plain-language purpose | When you need it |
| --- | --- | --- |
| `GUIDE.md` | This detailed explanation, including schemas and examples. | Understanding each file. |
| `train/` | Questions, positive/negative document addresses, and scores. | Training or inspecting examples. |
| `document_refs/` | An address book for the documents referenced by the release. | Looking up a document's ID, source file, and row. |
| `source_manifest.json` | The exact source version and ordered list of source files. | Turning a document number into a location. |
| `release_manifest.json` | Counts, release settings, model information, output-file lists, and checksums. | Understanding or checking the whole release. |
| `README.md` | The short dataset card, including the Hugging Face dataset configurations. | First introduction; loading `train` or `document_refs` on the Hub. |
| `finepdf_release.py` | Python helper for preparing the ordered source corpus and reading release rows. | Recommended training setup and selective text lookups. |
| `materialize_release_training.py` | A command that writes a separate text-containing subset using the resolver. | Preparing small question/positive/negative samples. |
| `release_data.py` | Shared column definitions and the writer used by the text-export command. | Required beside `materialize_release_training.py`. |
| `_SUCCESS` | A small receipt written when export completed successfully. | Checking that this is a completed release directory. |

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

{{retrieval_settings}}
The smallest observed negative count in this release is {{negative_minimum}}.

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
  "source_shard": "data/{{config}}/{{split}}/example-00001.parquet",
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
| `config` | Language/script selection, `{{config}}` here. |
| `split` | Source split, `{{split}}` here. |
| `revision` | Exact source commit, protecting the meaning of the document positions. |
| `num_rows` | Size of the entire source split, including unreferenced rows. |
| `shards` | Ordered source-file records containing `path`, `start_row`, and `num_rows`. |

This fictional example matches the document-120 explanation:

```json
{
  "repo_id": "{{repo}}",
  "config": "{{config}}",
  "split": "{{split}}",
  "revision": "{{revision}}",
  "num_rows": 200,
  "shards": [
    {"path": "data/{{config}}/{{split}}/example-00000.parquet", "start_row": 0, "num_rows": 100},
    {"path": "data/{{config}}/{{split}}/example-00001.parquet", "start_row": 100, "num_rows": 100}
  ]
}
```

The real manifest contains {{source_files}} files and {{source_rows}} rows. Its first source file
is `{{first_path}}`, starting at {{first_start}} with {{first_rows}} rows.

## `release_manifest.json`: the release inventory and history

This records what was exported and how to inspect it. Its main fields are:

| Field | Meaning |
| --- | --- |
| `format_version` | Version of the release layout, {{format_version}} for this release. |
| `references_only` | `true`: document text is stored upstream, rather than repeated in this release. |
| `partial` | Whether this was deliberately or actually an incomplete mining export; `{{partial_bool}}` here. |
| `missing_mining_shards` | Missing mining shard IDs; `{{missing_shards}}` here. |
| `query_count` | Primary plus additional queries, {{query_count}} here. |
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

The recorded pipeline configuration names these models:

{{models}}

Check `provenance.configuration_source` and its notes to distinguish verified
historical settings from repository defaults. A `null` model or code revision
means that revision was not recorded; it should not be inferred from filenames.

Long source documents can be shortened before a model reads them. The exact
generation, embedding, and reranker limits are recorded in `provenance`. The
document references still retrieve the full source text; the stored scores
remain those computed using the configured shortened views.

## `README.md` and `_SUCCESS`

`README.md` is the short dataset description shipped with the release. Its
opening block tells Hugging Face there are two loadable table configurations:
`train` and `document_refs`. This longer guide explains the data columns
and supplied helpers in more detail.

`_SUCCESS` is a tiny JSON file:

```json
{"format_version": {{format_version}}}
```

It marks a completed export. The reader checks for it before loading training
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
passages. The example below requests {{num_negatives}} negatives, which fits this
release's minimum of {{negative_minimum}}. If this minimum is zero, the example
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
manifest's relative paths, such as `data/{{config}}/{{split}}/...`.

```bash
python materialize_release_training.py \
  --release-dir . \
  --local-source-root /path/to/finepdfs \
  --output-dir /path/to/training-sample \
  --num-negatives {{num_negatives}} \
  --limit 1000
```

Choose a new output directory if that one already exists. Materializing many
examples can take substantial time and disk space. If no local source root is supplied,
the resolver downloads required Parquet files from the pinned source commit;
the number of examples does not cap download size.

For ordinary retrieval training, start with `train/`, the manifests, and
`load_source_dataset()` from the reading helper. Detailed generation and mining instructions will be available
in the code repository linked from the README.
