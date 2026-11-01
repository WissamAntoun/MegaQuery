{{front_matter}}

# MegaQuery-680M

We built this dataset of synthetic search queries and hard negatives from
[FinePDFs](https://huggingface.co/datasets/{{repo}}), for training text retrieval
models. Each example pairs a query with its source document and other documents
that look similar, together with relevance scores.

**Read more:** [Blog post](https://huggingface.co/blog/wissamantoun/megaquery-680m) ·
[Code and reproduction scripts](https://github.com/WissamAntoun/MegaQuery) ·
[Data guide and schemas](GUIDE.md)

The blog post explains the motivation and design choices.
The code repository includes generation, negative mining, and release preparation
scripts, with stage instructions in its docs directory.

## How we built it

We reimplement the **synthetic retrieval-data approach from
[Qwen3 Embedding](https://arxiv.org/html/2506.05176v3#A1)** with modifications
for FinePDFs. The idea is to start with a document and generate queries that
someone might use to find it, guided by a user persona, query type, difficulty,
and target length.

We use FinePDFs as our document corpus. Our pipeline supports different
query-generation models. {{query_configuration}}
We embed documents with
[`{{embedding_model}}`](https://huggingface.co/{{embedding_model}})
and retrieve negative candidates by cosine similarity to the **positive document**.
We use
[`{{reranker_model}}`](https://huggingface.co/{{reranker_model}})
to score the primary query against the positive document and each candidate.
We retain the retrieval and reranker scores so users can choose their own training
selection rules.

{{candidate_selection}}
We remove repeated negative references and candidates resolving to the positive
document. We include our model settings, document
truncation limits, and pipeline provenance in `release_manifest.json`.

## This release

| Item | Value |
| --- | --- |
| Language | {{language}} |
| Query groups | {{train_rows}} |
| Queries, including additional queries | {{query_count}} |
| Referenced documents | {{ref_rows}} |
| Negative documents per query group | {{negative_count_summary}} |
| FinePDFs source dataset | [{{repo}}](https://huggingface.co/datasets/{{repo}}) |
| FinePDFs language configuration / split | `{{config}}` / `{{split}}` |
| FinePDFs snapshot (commit) | `{{revision}}` |

## What is included

We provide two Parquet configurations:

- **`train`**: the primary query, its positive and negative document indices,
  aligned scores, generation metadata, and an `other_queries` list containing
  the additional queries and their metadata.
- **`document_refs`**: source document identifiers, source shard names, and row
  positions for the documents referenced by the training data.

Each row groups queries sharing a positive document. The reranker scores apply
**only to the primary `query`**. Additional queries can reuse the document
references for retrieval training. We order negative lists by descending retrieval score;
keep all score lists aligned with `negative_idxs` when selecting examples.

We reference document text in the pinned FinePDFs source and include manifests
and Python helpers to recover it without rerunning generation or mining.
See [GUIDE.md](GUIDE.md) for the full schemas and readable examples.

## Load the data

Install the reading dependencies with `pip install datasets huggingface_hub pyarrow`.

```python
from datasets import load_dataset

DATASET_ID = "wissamantoun/MegaQuery-680M"
train = load_dataset(DATASET_ID, "train", split="train", streaming=True)
row = next(iter(train))
print(row["query"], row["positive_idx"], row["negative_idxs"][:3])
```

### Recommended for training: prepare the source corpus once

Use `load_source_dataset()` to prepare the full source corpus for this language
as a locally cached, memory-mapped Arrow dataset. Its row positions match the
released document indices, so training can retrieve text directly without
building a Python ID lookup dictionary.

First, download the small helper files and manifests:

```python
import sys
from huggingface_hub import snapshot_download

release_dir = snapshot_download(
    DATASET_ID,
    repo_type="dataset",
    allow_patterns=[
        "source_manifest.json", "release_manifest.json", "_SUCCESS",
        "finepdf_release.py", "release_data.py",
        "materialize_release_training.py",
    ],
)
sys.path.insert(0, release_dir)
from finepdf_release import load_source_dataset

documents = load_source_dataset(release_dir).select_columns(["text"])

document_ids = [row["positive_idx"], *row["negative_idxs"][:8]]
texts = documents[document_ids]["text"]
print({"query": row["query"], "positive": texts[0], "negatives": texts[1:]})
```

The first preparation downloads the **entire source language split** and builds
the Arrow cache; allow time and disk space for both. Subsequent loads reuse the
cache. Memory mapping keeps the corpus on disk instead of loading all text into
RAM. You can choose the Arrow cache location with `cache_dir="/path/to/arrow-cache"`.
If you already have the pinned FinePDFs snapshot, pass
`local_source_root="/path/to/finepdfs"` to reuse those source files during preparation.

For a batch containing lists of source document indices, text lookup is:

```python
examples["documents"] = [
    documents[document_ids]["text"]
    for document_ids in examples["document_ids"]
]
```

**Document indices depend on the source order.** They are zero-based positions
in the pinned FinePDFs snapshot, concatenated in the shard order listed in
`source_manifest.json`. `load_source_dataset()` reproduces this order. Keep the
document corpus in that order and retain every row; shuffle or filter training
examples separately. Selecting only the `text` column preserves document positions.
Indices are not row numbers in `document_refs`.
If your training batch requires a fixed number of negatives, select training
rows with enough candidates; keep the document corpus intact.

### Inspect examples or prepare a small subset

Use `FinePDFResolver` for selective lookups without preparing the full Arrow
corpus. It reads source Parquet row groups as needed, which makes scattered
lookups less suitable for the inner loop of a large training run.

```python
from finepdf_release import FinePDFResolver, iter_training_examples

with FinePDFResolver(release_dir) as resolver:
    example = next(iter_training_examples([row], resolver, num_negatives={{num_negatives}}))
    print(example)  # query, positive, negative_0, ... text columns
```

The resolver downloads pinned source shards as needed, or accepts the same
`local_source_root` argument. A small sample with scattered negatives can still
require several large source shards.

For a text-only subset, use `materialize_release_training.py`; its usage
is explained in the guide. `all_queries=True` in the iterator expands additional
queries into separate examples without attaching the primary query's scores.

## Using the supervision

We generate queries and compute relevance scores automatically. Our mined hard
negatives can include other relevant documents, so you can filter candidates with
the retained scores. We do not apply an additional reranker-score threshold.
Avoid treating other queries sharing the same positive document as negatives
of each other when building training batches.

## License and attribution

We release this dataset under **{{license}}**. Source documents come from
[{{repo}}](https://huggingface.co/datasets/{{repo}}). See its dataset card
for source attribution and usage information. We build our data-generation recipe
on [Qwen3 Embedding](https://arxiv.org/abs/2506.05176).
