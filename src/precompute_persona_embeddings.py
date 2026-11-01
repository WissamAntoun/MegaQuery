#!/usr/bin/env python3
"""
Precompute L2-normalised persona embeddings using vLLM.

Reads a JSONL file where each line has a ``"persona"`` key, encodes all
personas with a vLLM embedding model, L2-normalises the vectors, and writes
the result to a NumPy ``.npy`` file that can be consumed directly by
``PersonaIndex`` in the main synth-gen pipeline.

Usage
-----
    python precompute_persona_embeddings.py \\
        --persona_path  /data/personas.jsonl \\
        --output_path   /data/persona_embs.npy \\
        --model         Qwen/Qwen3-Embedding-0.6B \\
        --batch_size    256 \\
        --max_model_len 512
"""

import argparse
import json
import logging
import os

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────── I/O ───────────────────────────────────────


def load_personas(path: str) -> list[str]:
    personas = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
            if "persona" not in obj:
                raise KeyError(f"Missing 'persona' key on line {lineno}: {line!r}")
            personas.append(obj["persona"])
    return personas


# ──────────────────────────────── Embedding ────────────────────────────────────


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


# Each query must come with a one-sentence instruction that describes the task
task = "Given a persona, retrieve relevant passages that are likely to interest the persona."


def embed_personas(
    personas: list[str],
    model: str,
    batch_size: int,
    max_model_len: int,
    tensor_parallel_size: int,
    trust_remote_code: bool,
) -> np.ndarray:
    """Embed all personas in batches and return a (N, D) float32 array."""
    from vllm import LLM

    logger.info("Loading vLLM embedding model: %s", model)
    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    logger.info("Embedding %d personas (batch_size=%d) …", len(personas), batch_size)

    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(personas), batch_size):
        batch = personas[start : start + batch_size]
        transformed_batch = [get_detailed_instruct(task, p) for p in batch]
        outputs = llm.embed(transformed_batch)
        batch_embs = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
        all_embeddings.append(batch_embs)
        logger.info(
            "  %d / %d processed", min(start + batch_size, len(personas)), len(personas)
        )

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, D)
    logger.info("Raw embedding matrix shape: %s", embeddings.shape)
    return embeddings


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return embeddings / norms


# ────────────────────────────────── Main ───────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute L2-normalised persona embeddings with vLLM"
    )
    p.add_argument(
        "--persona_path",
        required=True,
        help='JSONL file with one persona per line: {"persona": "..."}',
    )
    p.add_argument(
        "--output_path",
        required=True,
        help="Destination .npy file for the embedding matrix",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace model ID or local path for the vLLM embedding model",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of personas to embed per vLLM call",
    )
    p.add_argument(
        "--max_model_len",
        type=int,
        default=512,
        help="Maximum token length for the embedding model",
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    p.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="Disable trust_remote_code (enabled by default)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load personas ──────────────────────────────────────────────────────────
    logger.info("Loading personas from %s", args.persona_path)
    personas = load_personas(args.persona_path)
    logger.info("  %d personas loaded", len(personas))

    # ── Embed ──────────────────────────────────────────────────────────────────
    embeddings = embed_personas(
        personas=personas,
        model=args.model,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=not args.no_trust_remote_code,
    )

    # ── L2-normalise ───────────────────────────────────────────────────────────
    # normalised = l2_normalize(embeddings)
    normalised = embeddings  # Already normalised by the pooler_config in LLM
    logger.info(
        "Embedding matrix after normalisation: shape=%s  dtype=%s",
        normalised.shape,
        normalised.dtype,
    )

    # Sanity-check: norms should all be ≈ 1
    norms = np.linalg.norm(normalised, axis=1)
    logger.info(
        "Norm stats — min=%.6f  max=%.6f  mean=%.6f",
        norms.min(),
        norms.max(),
        norms.mean(),
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)
    np.save(args.output_path, normalised)
    logger.info("Saved embeddings to %s", args.output_path)


if __name__ == "__main__":
    main()
