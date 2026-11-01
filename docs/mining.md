# Compaction, document indexing and hard negatives

This guide adapts the embedding and hard-negative pipeline READMEs to the
completed v4 full-score path. The [stage launcher](../examples/run_stage.sh)
provides runnable commands with the released models and candidate counts.

## Compaction and canonical document shards

`compact_synth_parquets.py` groups generation records within each worker's
`--files-per-batch`. Its ID is the original `id` without the `__s1d_` suffix.
Query-specific fields become lists; repeated document fields use the first row.
It reduces repeated text storage, without promising global deduplication across
worker batches. Source query suffix IDs are discarded. Retain the raw Parquets
if you need those IDs later.

The public copy preserves null metadata positions and expands list-valued
queries during embedding aggregation. These compatibility fixes support the
compactor's actual output; [reproduction.md](reproduction.md) records the changes.

`ingest_parquet_embed_v3_shard.py` assigns sorted inputs round-robin to embedding
ranks. Each worker groups its own inputs by base document ID, preserves the
first document text, and collects distinct nonempty query texts in encounter
order. The first query becomes `repr_query_text`. No global query deduplication
or merging of groups across workers is performed.

Documents are embedded with `Qwen/Qwen3-Embedding-0.6B`. The example uses `text`
only and an empty prefix. `_truncate_text` uses the first 512 whitespace-separated
words with ` ...(truncated)` when cut, then caps the resulting string at 4,096
characters. The embedding model's token context is a separate limit (8,192 in
the example); `--embed-max-tokens` is a historical flag name for the word limit.

Each rank writes `embeddings/shard_XXXXX.parquet` and a manifest:

| Column | Purpose |
| --- | --- |
| `doc_id_local`, `doc_id_str` | Worker-local position and source identifier |
| `pos_text`, `pos_title` | Full retained document content |
| `all_query_texts`, `repr_query_text` | Ordered distinct queries and first query |
| `embedding` | Fixed-size float32 vector |

The manifest contains input identities, row/query counts and vector dimension.
These are run artifacts; the code repository does not include precomputed ones.

## USearch HNSW

`ingest_parquet_embed_v3_index.py` checks all rank manifests, row counts, input
coverage, and consistent dimensions. It assigns contiguous `doc_id_int` values
in embedding rank order, then row order. These integers identify mining slots;
they are **not public FinePDFs indices**.

The example selects cosine distance, f16 index vectors, connectivity 16,
construction expansion 128 and search expansion 64. These match the available
index launcher definitions; exact historical software builds and insertion
scheduling are not recovered. HNSW construction can vary with concurrency.

Outputs:

- `index.usearch`: document HNSW graph.
- `doc_id_map.parquet`: `doc_id_int` to `doc_id_str`.
- `doc_texts/shard_*.parquet` and `meta.json`: full content addressed by mining ID.
- `verification.json`: completeness result.

Use an explicit compact-input glob. Input paths must not include the embedding
or index outputs. Keep all embedding shards, their manifests, ID map and text
lookup consistent throughout mining/export. Build each index into a fresh work
root after changes to input files or embedding world size.

## Candidate retrieval

`generate_hard_negatives_v4.py` retrieves neighbors using each **positive
document's embedding**, not a query embedding. Set:

```text
--topk 64 --skip-top-n 32 --store-raw-reranker-scores
```

The first argument is the number **kept after skipping**. The search requests
96 non-self candidates (plus room to remove self), skips ranks 1–32, and scores
the remaining ranks 33–96. Small corpora and later alias removal can yield fewer
negatives. Skipping near duplicates is a heuristic, not a proof of irrelevance.
The retrieval score is `1 - cosine_distance`, a document-to-document similarity.
Temporary Parquets carry neighbors, queries, query IDs, and positive text into
reranking.

## Reranking and score contract

The completed path uses
`wissamantoun/bge-reranker-v2-gemma-seq-cls` through Ray Serve's `ScoreRequest`.
Each candidate is scored once using `repr_query_text`, also used for the
positive score. Additional queries share document references; they do not
receive independent reranker supervision.

The candidate view is title plus the first 500 words of text, including the
truncation marker where needed, capped at 3,000 characters and stripped. The
model context limit remains separately configurable.

With `--store-raw-reranker-scores`, the request disables score activation, retains
the returned raw scalar, and computes `sigmoid(raw)` for the ordinary score.
This assumes a one-label sigmoid reranker. It is not a calibrated probability.
Do not mix outputs from models or serving paths with a different score contract.

Intermediate mining Parquets retain the original field names `neg_embed`,
`neg_reranker`, `neg_reranker_raw`, `pos_reranker_score`,
`pos_reranker_raw_score`, `qids`, `query_texts`, `skipped_doc_ids` and
`doc_id_int`. Here `neg_embed` carries retrieval scores. The public export
normalizes the document addresses and exposes aligned `negative_retrieval_scores`
and `skipped_retrieval_scores`; see [data-guide.md](data-guide.md).

## Checkpoints and optional chunking

Each mining rank has its own checkpoint file and owns a round-robin subset of
embedding shards. Rerun the same configuration and output path to resume. Do not
reuse checkpoints after changing inputs, models or candidate selection settings.
`--pass2-chunk-size N` checkpoints completed reranking chunks separately.
If you enable chunking, consolidate completed shards before export:

```bash
python src/materialize_chunked_hard_negatives.py --input-dir work/eng_Latn/mining
```

It writes `shard_XXXXX/hard_negatives.parquet` plus `_MATERIALIZED_SUCCESS`.
The default examples use whole-shard reranking and need no consolidation.
Retain completed intermediate data until you verify the final export. No
reranker-score threshold is applied automatically.
