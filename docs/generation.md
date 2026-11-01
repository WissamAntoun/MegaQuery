# Persona retrieval and query generation

This guide adapts the synthetic generation README to the released recipe.
The implementation is [synth_gen_qwen3.py](../src/synth_gen_qwen3.py); its name
reflects the recipe's origin, while the release generator is Gemma.

## Inputs and persona retrieval

FinePDFs Parquets supply `id` and `text`. Source IDs are carried into generated
IDs; the helper does not invent replacement UUIDs. A persona JSONL supplies
`{"persona": "..."}` records. Its `.npy` matrix must preserve that exact row order.

[precompute_persona_embeddings.py](../src/precompute_persona_embeddings.py)
embeds each persona with `Qwen/Qwen3-Embedding-0.6B` using:

```text
Instruct: Given a persona, retrieve relevant passages that are likely to interest the persona.
Query:{persona}
```

Passages are embedded as plain text, without that instruction. Their view is
the first 5,000 characters with ` …(truncated)` appended if cut. The CPU persona
retriever L2-normalizes vectors and uses cosine similarity to retrieve the Top-1
profile. It uses the same profile for all four requested configurations.
A missing/invalid passage embedding produces empty persona candidates in the
retained implementation; inspect inference failures before a large run.

## Sampled configurations

The public path always samples two distinct query types and two distinct
difficulty levels uniformly without replacement, then takes their Cartesian
product. It makes four generation requests per input passage.

| Parameter | Values |
| --- | --- |
| Query types | `keywords`, `factual`, `conceptual`, `summary`, `yes_no`, `causal` |
| Difficulty | `high_school`, `college`, `phd` |
| Target words / probability | 10 / 15%, 20 / 30%, 30 / 45%, 50 / 10% |

A target length is sampled independently for each request. It is a prompt
request, not a strict output-length constraint. Repeated query text can merge
during canonical aggregation; the exporter preserves all matching generation
origins. `__s1d_0` through `__s1d_3` identify the requested variants. The historical
`stage1_*` metadata column names are retained for compatibility. The
`--disable_stage1` option is accepted; this public path always uses sampling.
The alternative LLM configuration-selection implementation is omitted.

## Generator, prompts and model input

The release uses `google/gemma-3-1b-it`. The retained request uses temperature
0.7, top-p 0.8, and maximum completion length 128 tokens. The example context
limit is 6,144 tokens. Both corpus and query language are English or French.
The prompt receives the bounded 5,000-character view described above.

The full system message, prompt, and query-type/difficulty definitions live in
`STAGE_TWO_SYSTEM`, `STAGE_TWO_PROMPT_TEMPLATE`, `QUESTION_TYPE_DEFINITIONS`, and
`DIFFICULTY_DEFINITIONS` in the generation module. They are also the prompts
described in the [blog](https://huggingface.co/blog/wissamantoun/megaquery-680m).
The prompt requests strict JSON containing a single `query` key, a natural query
from the persona's perspective, and paraphrasing rather than copied sentences.
The parser strips thinking tags and accepts a JSON object, a fenced JSON object,
or falls back to raw text. A model request is not a guarantee of valid output.

Ray runs passage embedding, materializes it, retrieves personas, deletes the
embedding Serve application, then starts the chat application. The CPU persona
path is the portable example. GPU persona retrieval remains available but can
need an additional GPU while the embedding deployment is still reserved.

## Output and recovery

Output Parquets retain original document fields plus `query`,
`stage1_character`, `stage1_question_type`, `stage1_difficulty`, `stage1_raw`,
`query_length_target`, `stage2_raw`, `persona_candidates`, and the expanded `id`.
All generation variants must complete before compaction or embedding. Preserve
raw outputs to retain original variant IDs and complete metadata.

`--checkpoint_path` uses Ray's development checkpoint API. The original sources
do not establish exactly-once recovery under all failure conditions. Use fresh
output paths when changing model, input files, rank partitions or prompts.
Sampling is unseeded, and inference/parallel scheduling can change output order;
the recipe is not a promise of identical query text.
