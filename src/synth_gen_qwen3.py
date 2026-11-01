#!/usr/bin/env python3
"""Retrieve a persona, sample four configurations, and generate retrieval queries.

Uses Ray Data, Ray Serve LLM, and vLLM. Inputs: FinePDFs Parquet (id, text),
persona JSONL, and a matching persona embedding matrix. Outputs: query Parquet.
"""

import argparse
import glob
import json
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────── Constants ─────────────────────────────────

QUESTION_TYPES = ["keywords", "factual", "conceptual", "summary", "yes_no", "causal"]
DIFFICULTIES = ["high_school", "college", "phd"]

QUESTION_TYPE_DEFINITIONS = {
    "keywords": "A short keyword or entity query (e.g., named entities, technical terms) directly mentioned in the passage.",
    "factual": "A question seeking a specific fact, date, number, or definition found in or derivable from the passage.",
    "conceptual": "A question asking for explanation, comparison, or deeper understanding of concepts in the passage.",
    "summary": "A request for the main idea, gist, or overview of the passage.",
    "yes_no": "A question answerable with yes or no strictly from the passage.",
    "causal": 'A "why" or "how" question about causes, mechanisms, or processes described in the passage.',
}

DIFFICULTY_DEFINITIONS = {
    "high_school": "Answerable with general high-school-level knowledge.",
    "college": "Requires undergraduate-level reasoning or domain knowledge.",
    "phd": "Requires graduate / expert-level depth or analytical reasoning.",
}

QUERY_LENGTH_WEIGHTED_LIST = [
    (10, 0.15),
    (20, 0.30),
    (30, 0.45),
    (50, 0.10),
]

# ─────────────────────────────────── Prompts ───────────────────────────────────



STAGE_TWO_SYSTEM = (
    "You are a search-query generator. You write realistic search queries "
    "from a character's perspective. Always respond with strict JSON only."
)

STAGE_TWO_PROMPT_TEMPLATE = """\
Given a character profile, a passage, and a set of requirements, write a \
realistic search query that this character would type or speak to find the passage.

## Guidelines
- The query must be written from the **character's perspective** and reflect \
their vocabulary and expertise level.
- The query must satisfy **all** of the requirements listed below.
- The query should be specific enough to retrieve the passage and not overly generic.
- Do NOT copy sentences verbatim from the passage; rephrase naturally.

## Output format (strict JSON — no extra keys)

{{"query": "<the generated query>"}}

## Task

The passage is written in **{corpus_language}**.
Generate the query in **{query_language}**.

**Character**:
{character}

**Requirements**:
- Question type: `{question_type}` — {question_type_definition}
- Difficulty: `{difficulty}` — {difficulty_definition}
- Target query length: approximately {length} words
- Output language: {query_language}

**Passage**:
{passage}

As a reminder, the task is to generate a search query following the guidelines and requirements below:
- The query must be written from the perspective of the following character: {character}
- The difficulty level of the query should be: {difficulty} — {difficulty_definition}
- The question type of the query should be: {question_type} — {question_type_definition}
- The target length of the query should be approximately {length} words.
- The query should be in {query_language}.
Generate the query in this strict JSON format: `{{"query": "<the generated query>"}}`.
"""

# ─────────────────────────────────── Helpers ───────────────────────────────────


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>…</think> reasoning blocks from the response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()




def parse_stage_two(text: str) -> str:
    """Parse Stage-2 JSON response; returns raw text on failure."""
    try:
        clean = text.strip()
        if clean.startswith("```"):
            parts = clean.split("```")
            clean = parts[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()
        result = json.loads(clean).get("query", text)
        return result if isinstance(result, str) else text
    except Exception:
        return text


def extract_generated_text(row: Dict[str, object]) -> str:
    """Extract generated text from either legacy or OpenAI-style LLM responses."""
    raw_text = row.get("generated_text")
    if isinstance(raw_text, str):
        return raw_text

    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                return text
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            maybe_text = item.get("text")
                            if isinstance(maybe_text, str):
                                parts.append(maybe_text)
                    if parts:
                        return "\n".join(parts)

    return ""


def sample_query_length() -> int:
    lengths, weights = zip(*QUERY_LENGTH_WEIGHTED_LIST)
    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()
    return int(np.random.choice(lengths, p=weights_arr))


def expand_stage1_disabled_rows(row: dict) -> List[dict]:
    """Expand one row into multiple rows without running Stage-1 LLM.

    Strategy:
      - Use the highest-ranked persona candidate from embedding retrieval.
      - Sample 2 question types and 2 difficulties (without replacement).
      - Return Cartesian-product combinations (up to 4 rows).
    """
    persona_candidates_raw = row.get("persona_candidates", "[]")
    top_character = ""
    try:
        persona_candidates = json.loads(persona_candidates_raw)
        if isinstance(persona_candidates, list) and persona_candidates:
            top_character = str(persona_candidates[0])
    except Exception:
        top_character = ""

    sampled_question_types = np.random.choice(QUESTION_TYPES, size=2, replace=False)
    sampled_difficulties = np.random.choice(DIFFICULTIES, size=2, replace=False)

    parent_id = row.get("__id__", "")
    expanded_rows: List[dict] = []
    variant_idx = 0
    for question_type in sampled_question_types:
        for difficulty in sampled_difficulties:
            out_row = dict(row)
            out_row["stage1_character"] = top_character
            out_row["stage1_question_type"] = str(question_type)
            out_row["stage1_difficulty"] = str(difficulty)
            out_row["stage1_raw"] = (
                '{"mode":"stage1_disabled","character":"'
                + top_character
                + '","question_type":"'
                + str(question_type)
                + '","difficulty":"'
                + str(difficulty)
                + '"}'
            )
            out_row["__id__"] = f"{parent_id}__s1d_{variant_idx}"
            out_row["id"] = f"{parent_id}__s1d_{variant_idx}"
            expanded_rows.append(out_row)
            variant_idx += 1

    return expanded_rows


# ─────────────────────────────── Persona Index ─────────────────────────────────


class PersonaIndex:
    """Persona JSONL + precomputed L2-normalised embeddings (.npy)."""

    def __init__(self, persona_path: str, embeddings_path: str):
        logger.info("Loading personas from %s", persona_path)
        self.personas: List[str] = []
        with open(persona_path) as f:
            for line in f:
                self.personas.append(json.loads(line)["persona"])
        logger.info("  %d personas loaded", len(self.personas))

        logger.info("Loading persona embeddings from %s", embeddings_path)
        embs = np.load(embeddings_path).astype(np.float32)
        norms = np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8, None)
        self.embeddings = embs / norms  # (N, D)  L2-normalised
        assert len(self.personas) == self.embeddings.shape[0]
        logger.info("  Embeddings shape: %s", self.embeddings.shape)

# ────────────────────────── Embed Processor Pre/Post ──────────────────────────


def make_embed_preprocess(embed_model: str):
    """Preprocess for the Serve deployment embed processor."""

    def preprocess(row: dict) -> dict:
        return {
            "method": "embeddings",
            "dtype": "EmbeddingCompletionRequest",
            "request_kwargs": {
                "model": embed_model,
                "input": row["__passage__"],
            },
            # carry-through all original columns
            **row,
        }

    return preprocess


def make_embed_postprocess():
    """Postprocess for the Serve deployment embed processor."""

    def postprocess(row: dict) -> dict:
        result = {
            k: v
            for k, v in row.items()
            if k not in {"method", "dtype", "request_kwargs", "data", "embeddings"}
        }

        embedding = row.get("embeddings")
        if embedding is None:
            data = row.get("data", [])
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    embedding = first.get("embedding")

        result["__passage_embedding__"] = embedding if embedding is not None else []
        return result

    return postprocess


# ──────────────────────────── Persona Retrieve Actor ───────────────────────────


class PersonaRetrieve:
    """Ray ``map_batches`` actor: cosine-similarity retrieval of top-k personas.

    The persona index embeddings are loaded once per actor replica.  When
    ``use_gpu=True`` the (B × N) similarity matrix is computed with PyTorch on
    the GPU assigned to this actor; otherwise plain NumPy is used.
    """

    def __init__(
        self,
        persona_path: str,
        persona_embeddings_path: str,
        top_k: int,
        use_gpu: bool = False,
    ):
        self.persona_index = PersonaIndex(persona_path, persona_embeddings_path)
        self.top_k = top_k
        self.use_gpu = use_gpu

        if use_gpu:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            # Pre-load the L2-normalised persona embeddings onto the GPU once.
            self.persona_tensor = torch.from_numpy(self.persona_index.embeddings).to(
                device
            )  # (N, D)
            logger.info("PersonaRetrieve: GPU matmul enabled on %s", device)
        else:
            logger.info("PersonaRetrieve: using NumPy CPU matmul")

    # ── internal helpers ────────────────────────────────────────────────────

    def _retrieve_gpu(self, embeddings: np.ndarray) -> List[List[str]]:
        import torch

        q = torch.from_numpy(embeddings).to(self.device)  # (B, D)
        # L2-normalise query vectors
        q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # (B, N) cosine similarities via batched matmul
        sims = q @ self.persona_tensor.T  # (B, N)
        if self.top_k == 1:
            indices = sims.argmax(dim=1).cpu().numpy()  # (B,)
            return [[self.persona_index.personas[int(i)]] for i in indices]
        sims_np = sims.cpu().numpy()
        return self._topk_from_sims(sims_np)

    def _retrieve_cpu(self, embeddings: np.ndarray) -> List[List[str]]:
        qnorms = np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
        q = embeddings / qnorms
        sims = q @ self.persona_index.embeddings.T  # (B, N)
        return self._topk_from_sims(sims)

    def _topk_from_sims(self, sims: np.ndarray) -> List[List[str]]:  # (B, N)
        if self.top_k == 1:
            indices = np.argmax(sims, axis=1)  # (B,) – faster than argpartition+argsort
            return [[self.persona_index.personas[int(i)]] for i in indices]
        results = []
        for i in range(sims.shape[0]):
            idx = np.argpartition(-sims[i], self.top_k)[: self.top_k]
            idx = idx[np.argsort(-sims[i, idx])]
            results.append([self.persona_index.personas[j] for j in idx])
        return results

    def _coerce_embedding(self, embedding: object) -> Optional[np.ndarray]:
        if embedding is None:
            return None

        try:
            array = np.asarray(embedding, dtype=np.float32)
        except Exception:
            return None

        if array.size == 0:
            return None

        array = array.reshape(-1)
        expected_dim = self.persona_index.embeddings.shape[1]
        if array.shape[0] != expected_dim:
            return None

        return array

    # ── callable ────────────────────────────────────────────────────────────

    def __call__(self, batch: Dict[str, list]) -> Dict[str, list]:
        raw_embs = batch["__passage_embedding__"]
        valid_indices: List[int] = []
        valid_embeddings: List[np.ndarray] = []
        invalid_shapes: List[object] = []

        for index, raw_embedding in enumerate(raw_embs):
            embedding = self._coerce_embedding(raw_embedding)
            if embedding is None:
                shape = getattr(raw_embedding, "shape", None)
                if shape is None:
                    try:
                        shape = np.asarray(raw_embedding).shape
                    except Exception:
                        shape = type(raw_embedding).__name__
                invalid_shapes.append(shape)
                continue

            valid_indices.append(index)
            valid_embeddings.append(embedding)

        candidates = [[] for _ in raw_embs]
        if valid_embeddings:
            embeddings = np.stack(valid_embeddings)  # (B_valid, D)

            if self.use_gpu:
                valid_candidates = self._retrieve_gpu(embeddings)
            else:
                valid_candidates = self._retrieve_cpu(embeddings)

            for index, row_candidates in zip(valid_indices, valid_candidates):
                candidates[index] = row_candidates

        if invalid_shapes:
            logger.warning(
                "PersonaRetrieve: skipped %d/%d rows with invalid embedding shapes. Examples: %s",
                len(invalid_shapes),
                len(raw_embs),
                invalid_shapes[:5],
            )

        batch["persona_candidates"] = [json.dumps(c) for c in candidates]
        # Remove the raw embedding column – it's no longer needed downstream
        del batch["__passage_embedding__"]

        # Columns that contain variable-length numpy object arrays (e.g.
        # per_page_languages: array(['eng_Latn', ...], dtype=object)) cannot be
        # converted to Arrow tensor format.  Convert them to plain Python lists
        # so PyArrow serialises them as list<string> instead.
        for key, col in batch.items():
            if key in ["per_page_languages"]:
                if isinstance(col, np.ndarray) and col.dtype == object:
                    batch[key] = [
                        v.tolist() if isinstance(v, np.ndarray) else v for v in col
                    ]

        return batch


# ────────────────────────── Pre/Post-process Builders ──────────────────────────






def make_s2_preprocess(
    text_field: str, corpus_language: str, query_language: str, model: str
):
    def preprocess(row: dict) -> dict:
        passage = row["__passage__"]
        query_length = sample_query_length()
        question_type = row["stage1_question_type"]
        difficulty = row["stage1_difficulty"]

        prompt = STAGE_TWO_PROMPT_TEMPLATE.format(
            passage=passage,
            character=row["stage1_character"],
            question_type=question_type,
            question_type_definition=QUESTION_TYPE_DEFINITIONS.get(question_type, ""),
            difficulty=difficulty,
            difficulty_definition=DIFFICULTY_DEFINITIONS.get(difficulty, ""),
            length=query_length,
            corpus_language=corpus_language,
            query_language=query_language,
        )

        request_extras = {"temperature": 0.7, "top_p": 0.8}
        if "qwen3" in model.lower():
            request_extras["presence_penalty"] = 1.5
            request_extras["temperature"] = 0.7
            request_extras["top_p"] = 0.8
            request_extras["top_k"] = 20
            request_extras["min_p"] = 0.0
            request_extras["repetition_penalty"] = 1.0

        return {
            # ── LLM inputs ──────────────────────────────────────
            "method": "chat",
            "dtype": "ChatCompletionRequest",
            "request_kwargs": {
                "model": model,
                "messages": [
                    {"role": "system", "content": STAGE_TWO_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": 128,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                **request_extras,
            },
            # ── Carry-through columns ─────────────────────────────
            "__id__": row["__id__"],
            "id": row["__id__"],
            "query_length_target": query_length,
            **{
                k: v
                for k, v in row.items()
                if k
                not in {
                    "__id__",
                    "id",
                    "__passage__",
                    "method",
                    "dtype",
                    "request_kwargs",
                    "generated_text",
                }
            },
        }

    return preprocess


def make_s2_postprocess():
    def postprocess(row: dict) -> dict:
        raw = strip_thinking(extract_generated_text(row))
        query = parse_stage_two(raw)

        result = {
            k: v
            for k, v in row.items()
            if k
            not in {
                "method",
                "dtype",
                "request_kwargs",
                "choices",
                "usage",
                "model",
                "object",
                "created",
                "generated_text",
                "__id__",
                "id",
                "__passage__",
            }
        }
        result["query"] = query
        result["stage2_raw"] = raw
        result["__id__"] = row["__id__"]
        result["id"] = row["__id__"]
        return result

    return postprocess


# ─────────────────────────────────── Main ──────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Ray Data + vLLM synthetic retrieval query generation"
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--input_glob",
        required=True,
        help='Glob pattern for input Parquet or JSONL files containing id and text',
    )
    p.add_argument("--output_path", required=True, help="Output directory (Parquet)")
    p.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional: directory for Ray Data checkpoint recovery",
    )

    # ── Persona ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--persona_path", required=True, help="JSONL file with persona strings"
    )
    p.add_argument(
        "--persona_embeddings_path",
        required=True,
        help="NumPy .npy file with precomputed persona embeddings",
    )
    p.add_argument("--top_k_personas", type=int, default=1)

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--text_field", default="text", help="Column name for the passage text"
    )
    p.add_argument("--language", choices=("English", "French"), default="English", help="Language of input passages")
    p.add_argument(
        "--query_language",
        choices=("English", "French"),
        default="English",
        help="Target language for generated queries",
    )

    # ── Sharding ──────────────────────────────────────────────────────────────
    p.add_argument("--rank", type=int, default=0, help="This worker's rank")
    p.add_argument("--world_size", type=int, default=1, help="Total number of workers")

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model", default="google/gemma-3-1b-it", help="vLLM chat model")
    p.add_argument(
        "--embed_model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="vLLM embedding model",
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor-parallel size for the chat LLM (number of GPUs per replica)",
    )
    p.add_argument(
        "--embed_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor-parallel size for the embedding model",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel LLM replicas per stage",
    )
    p.add_argument(
        "--embed_concurrency",
        type=int,
        default=1,
        help="Number of parallel embedding processor replicas",
    )
    p.add_argument(
        "--batch_size", type=int, default=64, help="LLM inference batch size"
    )
    p.add_argument(
        "--embed_batch_size",
        type=int,
        default=128,
        help="Embedding inference batch size",
    )
    p.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Max token context length for the chat LLM",
    )
    p.add_argument(
        "--embed_max_model_len",
        type=int,
        default=8192,
        help="Max token context length for the embedding model",
    )
    p.add_argument(
        "--persona_retrieve_on_gpu",
        action="store_true",
        default=False,
        help="Run persona cosine-similarity retrieval on GPU using PyTorch",
    )
    p.add_argument(
        "--disable_stage1",
        action="store_true",
        default=True,
        help=(
            "Retained CLI alias: the release always uses sampled configurations. Disable Stage 1 LLM. Instead, use top-1 retrieved persona and "
            "randomly sample 2 question types and 2 difficulties, expanding "
            "each input row into multiple rows with flat_map."
        ),
    )

    p.add_argument("--ray_object_store_memory_gb", type=float, default=None)
    p.add_argument("--cpus_per_replica", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    global ray, serve, ServeDeploymentProcessorConfig, build_processor
    global EmbeddingCompletionRequest, LLMConfig, ModelLoadingConfig, build_llm_deployment, ChatCompletionRequest
    import ray
    import ray.data
    from ray import serve
    from ray.data.llm import ServeDeploymentProcessorConfig, build_processor
    from ray.llm._internal.serve.core.configs.openai_api_models import (
        EmbeddingCompletionRequest,
    )
    from ray.serve.llm import LLMConfig, ModelLoadingConfig, build_llm_deployment
    from ray.serve.llm.openai_api_models import ChatCompletionRequest


    # ── Ray init (auto-uses all GPUs on the node / cluster) ───────────────────
    logger.info(
        "Starting Ray Data + Ray Serve LLM synthetic query generation with args: %s",
        args,
    )
    ray_address = os.environ.pop("RAY_ADDRESS", None)
    if ray_address:
        logger.warning(
            "RAY_ADDRESS was set to %s but will be ignored; starting a local cluster.",
            ray_address,
        )

    ray_dashboard_port = int(os.getenv("RAY_DASHBOARD_PORT", "8265"))
    ray_metrics_export_port = int(os.getenv("RAY_METRICS_EXPORT_PORT", "9090"))

    ray.init(
        address=None,
        ignore_reinit_error=True,
        **({"object_store_memory": int(args.ray_object_store_memory_gb * 1024**3)}
           if args.ray_object_store_memory_gb is not None else {}),
        _metrics_export_port=ray_metrics_export_port,
        dashboard_host="127.0.0.1",
        dashboard_port=ray_dashboard_port,
    )
    resources = ray.cluster_resources()
    logger.info("Ray initialised: %s", resources)

    logger.info(
        "Ray monitoring endpoints: dashboard=http://127.0.0.1:%d, metrics=http://127.0.0.1:%d/metrics",
        ray_dashboard_port,
        ray_metrics_export_port,
    )

    available_gpus = float(resources.get("GPU", 0.0))
    required_gpus = max(
        float(args.tensor_parallel_size), float(args.embed_tensor_parallel_size)
    )
    if args.persona_retrieve_on_gpu:
        required_gpus = max(required_gpus, 1.0)
    if available_gpus < required_gpus:
        raise RuntimeError(
            "Insufficient Ray GPU resources for requested config: "
            f"available GPU={available_gpus}, required at least {required_gpus}. "
            "Current run config requires GPUs for LLM stages."
        )

    # ── Checkpoint recovery (optional) ────────────────────────────────────────
    if args.checkpoint_path:
        from ray.data.checkpoint import CheckpointConfig

        ctx = ray.data.DataContext.get_current()
        ctx.checkpoint_config = CheckpointConfig(
            id_column="id",
            checkpoint_path=args.checkpoint_path,
            delete_checkpoint_on_success=False,
        )
        logger.info("Checkpoint recovery enabled at %s", args.checkpoint_path)

    # ── 1. Glob + rank-based file sharding ────────────────────────────────────
    all_files = sorted(glob.glob(args.input_glob, recursive=True))
    if not all_files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    is_jsonl = all(f.endswith(".jsonl") for f in all_files)
    is_parquet = all(f.endswith(".parquet") for f in all_files)
    if not (is_jsonl or is_parquet):
        raise ValueError(
            f"Input files must all be .jsonl or all be .parquet. Found files: {all_files}"
        )

    logger.info("Total files found: %d", len(all_files))

    # Assign files round-robin: rank 0 gets files[0, world_size, 2*world_size, …]
    my_files = all_files[args.rank :: args.world_size]
    logger.info(
        "Rank %d/%d → processing %d files", args.rank, args.world_size, len(my_files)
    )

    if not my_files:
        raise ValueError(
            f"Rank {args.rank} has no files to process after sharding. Check your glob pattern and world_size."
        )

    if is_jsonl:
        ds = ray.data.read_json(
            my_files,
            # lines=True,
            override_num_blocks=64,
        )
    elif is_parquet:
        ds = ray.data.read_parquet(
            my_files,
            override_num_blocks=64,
        )
    else:
        raise ValueError(
            "Unsupported file format. Only .jsonl and .parquet are supported."
        )


    # ── 3. Add passage alias column ──────────────────────────────────────────
    def add_passage_alias(row: dict) -> dict:
        passage = str(row[args.text_field])
        # truncate to 5000 chars with (... truncated) suffix
        if len(passage) > 5000:
            passage = passage[:5000] + " …(truncated)"
        row["__passage__"] = passage
        row["__id__"] = row["id"]
        return row

    ds = ds.map(add_passage_alias)

    # ── 4. Build and run Ray Serve embedding deployment ───────────────────────
    embed_app_name = f"synth-gen-embed-app-r{args.rank}"
    embed_deployment_name = f"synth-gen-embed-deployment-r{args.rank}"
    embed_llm_config = LLMConfig(
        model_loading_config=ModelLoadingConfig(
            model_id=args.embed_model,
            model_source=args.embed_model,
        ),
        deployment_config=dict(
            name=embed_deployment_name,
            autoscaling_config=dict(
                min_replicas=args.embed_concurrency,
                max_replicas=args.embed_concurrency,
            ),
        ),
        engine_kwargs={
            "max_model_len": args.embed_max_model_len,
            "tensor_parallel_size": args.embed_tensor_parallel_size,
            "gpu_memory_utilization": 0.9,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "enforce_eager": True,
            "trust_remote_code": True,
            "max_num_batched_tokens": 8192,
            "runner": "pooling",
            "disable_log_stats": False,  # emit token-throughput stats
        },
        placement_group_config={
            "bundles": [{"GPU": 1, "CPU": args.cpus_per_replica}] * args.embed_tensor_parallel_size,
            "strategy": "PACK",
        },
        runtime_env=dict(
            env_vars={
                "VLLM_DISABLE_COMPILE_CACHE": "1",
                "VLLM_LOGGING_LEVEL": "INFO",
            },
        ),
    )

    embed_llm_app = build_llm_deployment(
        embed_llm_config,
        name_prefix=f"synth-gen-embed-r{args.rank}",
        override_serve_options={"name": embed_deployment_name},
    )
    serve.run(
        embed_llm_app,
        name=embed_app_name,
        route_prefix=f"/{embed_deployment_name}",
        logging_config={"enable_access_log": False},
    )

    embed_processor_config = ServeDeploymentProcessorConfig(
        deployment_name=embed_deployment_name,
        app_name=embed_app_name,
        dtype_mapping={"EmbeddingCompletionRequest": EmbeddingCompletionRequest},
        concurrency=args.embed_concurrency,
        batch_size=args.embed_batch_size,
        should_continue_on_error=True,
    )

    embed_processor = build_processor(
        embed_processor_config,
        preprocess=make_embed_preprocess(args.embed_model),
        postprocess=make_embed_postprocess(),
    )

    ds = embed_processor(ds)

    logger.info("Materializing embedding phase before persona retrieval ...")
    ds = ds.materialize()
    logger.info("Embedding phase finished.")

    # ── 5. Cosine-similarity retrieval of top-k persona candidates ────────────
    #    Runs after the embedding phase has fully completed so the pipeline is
    #    explicitly staged as: embed -> retrieve -> delete embed app -> chat.
    #    When --persona_retrieve_on_gpu is set the actor requests 1 GPU and
    #    uses PyTorch for the (B × N) matmul; otherwise it uses NumPy on CPU.
    retrieve_num_gpus = 1 if args.persona_retrieve_on_gpu else 0
    ds = ds.map_batches(
        PersonaRetrieve,
        fn_constructor_kwargs={
            "persona_path": args.persona_path,
            "persona_embeddings_path": args.persona_embeddings_path,
            "top_k": args.top_k_personas,
            "use_gpu": args.persona_retrieve_on_gpu,
        },
        batch_size=args.embed_batch_size,
        num_gpus=retrieve_num_gpus,
        compute=ray.data.ActorPoolStrategy(size=args.concurrency),
        num_cpus=4,
    )

    logger.info("Materializing persona retrieval phase before chat startup ...")
    ds = ds.materialize()
    logger.info("Persona retrieval phase finished.")

    logger.info("Deleting embedding Serve app %s ...", embed_app_name)
    serve.delete(embed_app_name, _blocking=True)
    logger.info("Embedding Serve app deleted.")

    extra_engine_kwargs = {}
    if "qwen3" in args.model.lower():
        extra_engine_kwargs["reasoning_parser"] = "qwen3"
        extra_engine_kwargs["limit_mm_per_prompt"] = {
            "image": 0,
            "video": 0,
        }
    elif "gemma" in args.model.lower():
        extra_engine_kwargs["block_size"] = 64

    # ── 6. Build and run the query-generation chat deployment ───────────────
    llm_app_name = f"synth-gen-chat-app-r{args.rank}"
    llm_deployment_name = f"synth-gen-chat-deployment-r{args.rank}"
    llm_deploy_config = LLMConfig(
        model_loading_config=ModelLoadingConfig(
            model_id=args.model,
            model_source=args.model,
        ),
        deployment_config=dict(
            name=llm_deployment_name,
            autoscaling_config=dict(
                min_replicas=args.concurrency,
                max_replicas=args.concurrency,
            ),
        ),
        engine_kwargs={
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": 0.9,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "trust_remote_code": True,
            "disable_log_stats": False,  # emit token-throughput stats
            "max_num_batched_tokens": 16384,
            **extra_engine_kwargs,
        },
        placement_group_config={
            "bundles": [{"GPU": 1, "CPU": args.cpus_per_replica}] * args.tensor_parallel_size,
            "strategy": "PACK",
        },
        runtime_env=dict(
            env_vars={
                "VLLM_DISABLE_COMPILE_CACHE": "1",
                "VLLM_LOGGING_LEVEL": "INFO",
            },
        ),
    )

    llm_app = build_llm_deployment(
        llm_deploy_config,
        name_prefix=f"synth-gen-chat-r{args.rank}",
        override_serve_options={"name": llm_deployment_name},
    )
    serve.run(
        llm_app,
        name=llm_app_name,
        route_prefix=f"/{llm_deployment_name}",
        logging_config={"enable_access_log": False},
    )

    llm_processor_config = ServeDeploymentProcessorConfig(
        deployment_name=llm_deployment_name,
        app_name=llm_app_name,
        dtype_mapping={"ChatCompletionRequest": ChatCompletionRequest},
        concurrency=args.concurrency * 8,
        batch_size=args.batch_size,
        should_continue_on_error=True,
    )

    ds = ds.flat_map(expand_stage1_disabled_rows)

    # ── 8. Stage 2: query generation ─────────────────────────────────────────
    processor_s2 = build_processor(
        llm_processor_config,
        preprocess=make_s2_preprocess(
            args.text_field, args.language, args.query_language, args.model
        ),
        postprocess=make_s2_postprocess(),
    )
    ds = processor_s2(ds)

    # ── 9. Write output ───────────────────────────────────────────────────────
    os.makedirs(args.output_path, exist_ok=True)
    logger.info("Writing results to %s", args.output_path)
    ds.write_parquet(args.output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
