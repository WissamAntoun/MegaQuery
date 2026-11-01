# Environment and hardware

## CPU tools and readers

Use Linux and Python 3.10 or newer. Python 3.12 is recommended for a GPU
installation; select a version supported by your accelerator's vLLM build.
From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHONDONTWRITEBYTECODE=1 python -m pytest
```

The CPU requirements cover Parquet/Arrow, DuckDB, source loading, USearch,
progress reporting, export, and fixtures. USearch is pinned to a wheel version
used for CPU validation, rather than claimed as the historical build.
Requirements ranges describe supported installation candidates, not a recovered
historical lockfile. The validation environment is recorded in [reproduction.md](reproduction.md).

## GPU inference

Generation and mining use Ray Data, Ray Serve LLM and vLLM. Install PyTorch and
vLLM using builds compatible with your GPU, driver and Python version, then
install the remaining requirements. A CPU vLLM build is not a replacement for
these GPU stages.

The retained serving code imports these specific interfaces:

- `ray.data.llm.ServeDeploymentProcessorConfig` and `build_processor`.
- `ray.serve.llm.LLMConfig`, `ModelLoadingConfig`, `build_llm_deployment`.
- `EmbeddingCompletionRequest`, `ScoreRequest`, `ChatCompletionRequest`.
- vLLM `LLM.embed()` and the `pooling` runner.

Some request types are in Ray's **private API**. An arbitrary stable Ray version
is not sufficient. The available build definitions used Ray `3.0.0.dev0` wheels;
one recorded public development commit is
`36a5d61713e98f7976855a0179dc31e9ed84c98d`. Matching wheel example for CPython 3.12,
Linux x86-64:

```bash
python -m pip install 'ray[data,serve,llm]==3.0.0.dev0' \
  -f https://s3-us-west-2.amazonaws.com/ray-wheels/master/36a5d61713e98f7976855a0179dc31e9ed84c98d/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl
python -m pip install -r requirements-gpu.txt
python examples/check_runtime.py
```

The original launch definitions name vLLM 0.17 for generation and 0.19 for
reranking; another available build definition uses 0.18. These definitions do
not establish the exact packages loaded by every completed job. Use separate
environments for generation and mining if necessary. The GPU stack has not been
validated in this code-release preparation. Passing the interface check verifies
imports only; run a small GPU shard before committing to a full run. Record your
resolved packages, model commits and code commit with your new run.

The published sequence-classification reranker can be loaded directly. Optional
`src/convert_model_to_seq_cls.py` retains the vLLM conversion utility for a new
model conversion and needs Transformers, PyTorch and Accelerate. It does not
establish the exact weights of a previous conversion.

Model downloads may require accepting the provider's model terms. Use normal
Hugging Face authentication outside this repository. `HF_HOME` and
`HF_HUB_CACHE` can select cache locations. Model flags accept Hub IDs or local
model directories. No credentials are required in configuration files.

## Resource planning

- **Persona embeddings / document embeddings / generation:** at least one
  compatible GPU per model replica. Start with concurrency 1 and tensor parallel
  size 1. Replica GPU count is `concurrency × tensor_parallel_size`. Embedding
  and chat deployments run in separate phases. CPU persona retrieval avoids
  reserving another GPU while the embedding deployment is still live.
- **Memory:** generation materializes each worker's entire intermediate dataset;
  embedding aggregation and its vector matrix also reside in host RAM. Increase
  worker count and reduce files per worker to fit memory. Reducing the encoding
  batch size only reduces inference pressure; it does not bound aggregation RAM.
  The portable generation example reserves 4 CPUs per model worker. Plan extra
  CPUs for Ray and data actors; these are scheduling requests, not benchmarked
  minimum hardware specifications.
- **Reranking:** the Gemma-based reranker is substantially larger than the 1B
  generator. Its weights, KV/activation buffers, batch sizes and context length
  must fit GPU memory; increase tensor parallel size or reduce batches. No fixed
  minimum VRAM or throughput has been measured for this portable release.
- **Indexing:** CPU only. Raw vectors need `N × D × bytes_per_element` bytes,
  before HNSW links, keys, text lookup, Parquet decoding and construction overhead.
  At 100M vectors and dimension 1024, f16 vector storage alone is about 191 GiB;
  f32 is about 381 GiB. Full-language runs need a large-memory machine.
- **Retrieval:** the retained v4 implementation calls `Index.load()`, loading the
  index into RAM. Document text lookups use memory-mapped Arrow files. Budget
  index RAM for each independent mining process.
- **Export:** CPU only, with configurable DuckDB memory/spill paths, thread count,
  and training workers. Leave additional RAM beyond the DuckDB limit for Arrow,
  NumPy, worker processes and the OS.
- **Disk:** allow space for the complete pinned FinePDFs split, model weights,
  raw generation, compacted queries, float32 embedding Parquets, HNSW, document
  text Parquet and Arrow copies, mining outputs, export working files and final
  Parquets. Full runs require substantial storage, potentially terabytes.

All workers require access to their configured input/output paths. The examples
start local Ray instances; run independent ranks on separate machines or give
each process its own ports and scratch space. No scheduler, container runtime,
monitoring service or site module is required by the examples.
