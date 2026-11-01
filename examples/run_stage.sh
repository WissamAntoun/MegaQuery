#!/usr/bin/env bash
# Run from the repository root. Each stage writes only under configured work paths.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'Usage: bash examples/run_stage.sh CONFIG.env STAGE' >&2
  exit 2
fi
set -a
source "$1"
set +a
stage=$2
case "$CONFIG:$LANGUAGE" in eng_Latn:English|fra_Latn:French) ;; *) echo 'Choose an English or French configuration' >&2; exit 2 ;; esac
PYTHON=${PYTHON:-python}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
source_glob="$FINEPDF_ROOT/data/$CONFIG/train/*.parquet"
query_glob="$WORK_ROOT/queries/**/*.parquet"
compact_glob="$WORK_ROOT/compact/*.parquet"
ray_args=()
if [[ -n ${RAY_OBJECT_STORE_MEMORY_GB:-} ]]; then
  ray_args=(--ray_object_store_memory_gb "$RAY_OBJECT_STORE_MEMORY_GB")
fi
case "$stage" in
  prepare)
    "$PYTHON" src/prepare_source.py --release-dir "$SOURCE_RELEASE" --config "$CONFIG" --output-dir "$FINEPDF_ROOT"
    ;;
  personas)
    "$PYTHON" src/precompute_persona_embeddings.py --persona_path "$PERSONA_PATH" \
      --output_path "$PERSONA_EMBEDDINGS" --model "$EMBED_MODEL" --max_model_len 512
    ;;
  generate)
    "$PYTHON" src/synth_gen_qwen3.py --input_glob "$source_glob" \
      --output_path "$WORK_ROOT/queries/$GEN_RANK" \
      --persona_path "$PERSONA_PATH" --persona_embeddings_path "$PERSONA_EMBEDDINGS" \
      --model "$GEN_MODEL" --embed_model "$EMBED_MODEL" \
      --language "$LANGUAGE" --query_language "$LANGUAGE" --top_k_personas 1 --disable_stage1 \
      --rank "$GEN_RANK" --world_size "$GEN_WORLD_SIZE" \
      --concurrency "$CONCURRENCY" --embed_concurrency "$EMBED_CONCURRENCY" \
      --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" --embed_tensor_parallel_size 1 \
      --cpus_per_replica "$CPUS_PER_REPLICA" --batch_size "$GEN_BATCH_SIZE" \
      --embed_batch_size "$PASSAGE_EMBED_BATCH_SIZE" --max_model_len 6144 --embed_max_model_len 8192 \
      "${ray_args[@]}"
    ;;
  compact)
    "$PYTHON" src/compact_synth_parquets.py --glob "$query_glob" --output-dir "$WORK_ROOT/compact" \
      --query-id-field id --num-workers "$COMPACT_WORKERS" --files-per-batch "$COMPACT_FILES_PER_BATCH"
    ;;
  embed)
    "$PYTHON" src/ingest_parquet_embed_v3_shard.py --glob "$compact_glob" \
      --fields text --query-field query --id-field id --embed-model "$EMBED_MODEL" \
      --embed-max-model-len 8192 --embed-max-tokens 512 --encode-batch-size "$ENCODE_BATCH_SIZE" \
      --output-dir "$WORK_ROOT/index" --rank "$EMBED_RANK" --world-size "$EMBED_WORLD_SIZE"
    ;;
  index)
    "$PYTHON" src/ingest_parquet_embed_v3_index.py --glob "$compact_glob" \
      --output-dir "$WORK_ROOT/index" --world-size "$EMBED_WORLD_SIZE" \
      --metric cos --dtype f16 --connectivity 16 --expansion-add 128 --expansion-search 64
    ;;
  mine)
    mine_ray_args=()
    if [[ -n ${RAY_OBJECT_STORE_MEMORY_GB:-} ]]; then
      mine_ray_args=(--ray-object-store-memory-gb "$RAY_OBJECT_STORE_MEMORY_GB")
    fi
    "$PYTHON" src/generate_hard_negatives_v4.py --embeddings-dir "$WORK_ROOT/index/embeddings" \
      --index-dir "$WORK_ROOT/index" --output-dir "$WORK_ROOT/mining" \
      --rank "$MINE_RANK" --world-size "$MINE_WORLD_SIZE" \
      --topk 64 --skip-top-n 32 --search-threads "$THREADS" \
      --reranker-model "$RERANKER_MODEL" --reranker-concurrency "$RERANKER_CONCURRENCY" \
      --reranker-tp-size "$RERANKER_TP_SIZE" --reranker-batch-size 64 \
      --store-raw-reranker-scores "${mine_ray_args[@]}"
    ;;
  export)
    "$PYTHON" src/export_finepdf_release.py --index-dir "$WORK_ROOT/index" \
      --hard-negatives-dir "$WORK_ROOT/mining" --queries-glob "$compact_glob" \
      --finepdf-root "$FINEPDF_ROOT" --finepdf-glob "$source_glob" \
      --source-config "$CONFIG" --source-revision "$SOURCE_REVISION" \
      --expected-source-manifest "$SOURCE_RELEASE/data/$CONFIG/source_manifest.json" \
      --provenance-json "$PROVENANCE" --output-dir "$WORK_ROOT/release" \
      --work-dir "$WORK_ROOT/export-work" --threads "$THREADS" \
      --train-workers "$TRAIN_WORKERS" --memory-limit "$EXPORT_MEMORY"
    ;;
  *) echo 'Stages: prepare personas generate compact embed index mine export' >&2; exit 2 ;;
esac
