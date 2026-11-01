# Reproduction scope

The public [blog](https://huggingface.co/blog/wissamantoun/megaquery-680m), dataset
card, per-language manifests and packaged readers describe the completed English
and French release. The included scripts preserve the corresponding generation,
v3 embedding/indexing, and v4 full-score mining/export algorithms.

## Recorded recipe

| Stage | Public recipe |
| --- | --- |
| Source | FinePDFs `eng_Latn` / `fra_Latn`, train, commit `220bac3acbf07789502c621d2d33952f51ac7f86` |
| Persona / document embeddings | `Qwen/Qwen3-Embedding-0.6B` |
| Configuration | Top-1 retrieved persona; two types × two difficulties; sampled target length |
| Generation | `google/gemma-3-1b-it`; first 5,000 characters with marker |
| Embedding view | First 512 words with marker, capped at 4,096 characters |
| Retrieval | Positive-document vector; 96 non-self neighbors; skip 32; keep 64 |
| Reranking | `wissamantoun/bge-reranker-v2-gemma-seq-cls`; first stored query |
| Reranker document view | Title + first 500 words with marker, capped at 3,000 characters |
| Scores | Document retrieval similarity; raw reranker scalar and sigmoid(raw) |
| Export | Language-local positions in pinned source order; aligned score lists |

The configuration JSON and stage launcher make a coherent **new run** possible.
Resource settings are adjustable examples. A model or code revision set to null
means it was not recorded. Do not substitute a current model revision and label
it a verified historical revision.

## Remaining gaps for exact regeneration

- The historical persona JSONL, its source/version and matching vector file are
  not included. You must supply and attribute a persona collection and compute
  its embeddings. A substitute changes the generated queries.
- Exact model/tokenizer revisions, historical code/prompt revisions and all
  job-level overrides are not fully recoverable from the available materials.
- No complete historical environment lockfile is available. The serving APIs
  depend on a Ray development build and accelerator-specific vLLM. GPU inference
  has not been rerun during this code-release preparation.
- Sampling has no recorded global seed. Inference scheduling, file partitions,
  group order and concurrent HNSW insertion can affect outputs. Different worker
  partitions can change mining groups even when source indices remain valid.
- The full source, generation Parquets, persona vectors, embedding shards,
  HNSW graph and full-score mining artifacts must be downloaded or recomputed.
  They are not part of the code repository. There is no claimed way to recreate
  the exact historical artifact bytes from these scripts alone.

To **use the published training dataset**, none of the generation artifacts are
needed. Download its annotation shards and manifests, then use the packaged
readers against the pinned FinePDFs source.

## Public-copy implementation changes

These changes apply only to this code distribution and future runs:

- Configuration sampling is the only generation path; the optional LLM selector
  is omitted. Prompts and sampling distributions are retained.
- Generation no longer reserves four GPUs and 500 GB of object-store memory.
  Replica counts, CPU requests and optional object-store limits are configurable.
  Tensor parallel size maps to tensor parallelism, with one GPU scheduling bundle
  per device. Ray manages visible devices; no accelerator-specific mask is forced.
- GPU serving imports are deferred so CLI help and CPU helpers work independently.
- Compaction keeps null metadata slots aligned. Embedding aggregation expands
  compacted query lists rather than converting a complete list into one string.
  These fixes can change new outputs relative to affected older inputs; they are
  not changes to the published dataset.
- The public format emits retrieval-score names. Original intermediate mining
  fields remain compatible with the copied v4/export path.
- Retrieval handles USearch's singleton-query result shape as well as batch results.
- Source preparation checks a supplied manifest, and export can require exact
  agreement with its file order/counts. The new two-language assembler preserves
  each language's numbering and inventories.

CPU fixture validation covers generation prompt/sampling helpers, compaction,
canonical query aggregation, real USearch indexing/retrieval, source remapping,
score alignment/cleanup, export checksums and both combined-layout readers.
This validates data transformations and interfaces, not full-scale inference.

## CPU validation environment

Fixtures were run with Python 3.10.20, NumPy 2.2.6, PyArrow 25.0.1, DuckDB 1.4.3,
Datasets 4.8.5, USearch 2.23.0 and pytest 9.1.1. This is the code-release test
environment, not the environment used for historical generation or mining.
