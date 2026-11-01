"""Prepare a public dataset card and guide; retain diagnostics outside the release.

Also works on a completed release without rescanning training data or source text.
Replaced files and detailed reports are archived outside the published directory.
"""

import argparse
import copy
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import uuid


LOG = logging.getLogger(__name__)


def canonical_file_label(value):
    """A logical mining-artifact label for internal matching reports."""
    if not value:
        return value
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    namespace = next((part for part in reversed(parts[:-1])
                      if part in ("doc_texts", "embeddings")), "canonical_text")
    return namespace + "/" + parts[-1]


def public_manifest(manifest):
    result = copy.deepcopy(manifest)
    # Scientific settings and file inventories stay public; run diagnostics do not.
    for key in ("source_identity_resolution", "negative_alias_resolution", "generation_inputs",
                "internal_report_dir", "query_group_exclusion"):
        result.pop(key, None)
    for section in ("generation", "mining", "reranker"):
        result.get("provenance", {}).get(section, {}).pop("configured_by", None)
    result["data_processing"] = {
        **result.get("data_processing", {}),
        "document_indices": "Zero-based positions in the pinned source manifest's shard order.",
        "negative_cleanup": "Repeated negative references and references to the positive are removed.",
        "negative_order": "Descending retrieval score; all negative score lists are aligned with negative_idxs.",
    }
    cleanup = manifest.get("negative_alias_resolution", {})
    if cleanup.get("removed_empty_negatives") or cleanup.get("removed_empty_skipped"):
        result["data_processing"]["empty_candidates"] = (
            "Empty candidate artifacts without source references are removed.")
    exclusions = manifest.get("query_group_exclusion", {})
    if exclusions.get("excluded_groups"):
        result["data_processing"]["query_cleanup"] = (
            "Query groups containing serialized byte-string query artifacts are excluded.")
    return result


def template_values(manifest, source):
    configs = manifest["configs"]
    provenance = manifest.get("provenance", {})
    generation = provenance.get("generation", {})
    mining = provenance.get("mining", {})
    language, language_code = {
        "fra_Latn": ("French", "fr"), "eng_Latn": ("English", "en"),
    }.get(source["config"], (source["config"], None))
    def count(value):
        return format(value, ",d")
    values = {
        "format_version": manifest["format_version"],
        "language": language, "config": source["config"], "split": source["split"],
        "repo": source["repo_id"], "revision": source["revision"],
        "license": provenance.get("license", "not recorded"),
        "embedding_model": mining.get("model", "not recorded"),
        "reranker_model": provenance.get("reranker", {}).get("model", "not recorded"),
        "train_rows": count(configs["train"]["num_rows"]),
        "query_count": count(manifest["query_count"]),
        "ref_rows": count(configs["document_refs"]["num_rows"]),
        "source_rows": count(source["num_rows"]),
        "train_files": count(len(configs["train"]["shards"])),
        "ref_files": count(len(configs["document_refs"]["shards"])),
        "source_files": count(len(source["shards"])),
        "negative_counts": ", ".join(map(str, manifest["negative_counts"])) or "none",
        "partial": "Yes" if manifest["partial"] else "No",
        "partial_bool": str(manifest["partial"]).lower(),
        "complete": "No (partial release)" if manifest["partial"] else "Yes",
        "missing_shards": json.dumps(manifest.get("missing_mining_shards", [])),
        "num_negatives": min([8] + manifest["negative_counts"]),
        "negative_minimum": str(min(manifest["negative_counts"], default=0)),
    }
    counts = manifest["negative_counts"]
    values["negative_count_summary"] = (
        "{}–{}, depending on the group".format(min(counts), max(counts))
        if counts and min(counts) != max(counts) else str(counts[0]) if counts else "none")
    first = source["shards"][0]
    values.update(first_path=first["path"], first_start=count(first["start_row"]),
                  first_rows=count(first["num_rows"]))
    values["models"] = "\n".join(
        "- {}: `{}`.".format(label, provenance.get(section, {}).get("model", "not recorded"))
        for section, label in (("generation", "Generation model"), ("mining", "Embedding model"),
                               ("reranker", "Reranker")))
    kept, skipped = mining.get("topk_kept"), mining.get("skip_top_n")
    values["candidate_selection"] = (
        "We retrieve the Top-k = {} non-self candidates, skip the first {} ranks, "
        "and rerank the remaining {} candidates.".format(kept + skipped, skipped, kept)
        if isinstance(kept, int) and isinstance(skipped, int) else
        "Our retrieval and reranking settings are in the release manifest.")
    values["retrieval_settings"] = (
        values["candidate_selection"] + " Retrieval uses the positive document's embedding; "
        "reranking uses the primary question. Removing repeated negative references can reduce that count.")
    values["query_configuration"] = (
        "We use the most relevant retrieved persona and randomly sample query types, "
        "difficulty levels, and target lengths to generate several variants per document." if generation.get("stage1_enabled") is False
        and generation.get("top_k_personas") == 1 else
        "We generate queries using a persona, query type, difficulty level, and target length. "
        "We include these settings in the release manifest.")
    front = ["---", "license: " + json.dumps(values["license"])]
    if language_code:
        front.extend(["language:", "  - " + language_code])
    front.extend([
        "task_categories:", "  - text-retrieval", "tags:", "  - synthetic", "  - hard-negatives",
        "  - finepdfs", "  - qwen3-embedding", "configs:",
        "  - config_name: train", "    data_files:", "      - split: train", "        path: train/*.parquet",
        "  - config_name: document_refs", "    data_files:", "      - split: train",
        "        path: document_refs/*.parquet", "---",
    ])
    values["front_matter"] = "\n".join(front)
    return values


def render_template(name, manifest, source):
    template = Path(__file__).with_name(name).read_text(encoding="utf-8")
    values = template_values(manifest, source)
    return re.sub(r"\{\{([a-z_]+)\}\}", lambda match: str(values[match.group(1)]), template)


def render_guide(root, manifest, source):
    return render_template("release_guide_template.md", manifest, source)


def render_readme(manifest, source):
    return render_template("release_readme_template.md", manifest, source)


def prepare_publication(root, backup_dir=None):
    """Archive internal reports and write only user-facing metadata and docs.

    backup_dir must be new and outside the published directory. It retains
    replaced files and original reports, including their private provenance.
    """
    root = Path(root).resolve()
    backup_dir = Path(backup_dir).resolve() if backup_dir is not None else root.with_name(
        root.name + ".internal-" + uuid.uuid4().hex[:12])
    if backup_dir == root or root in backup_dir.parents:
        raise ValueError("Backups must be outside the release directory")
    manifest_path = root / "release_manifest.json"
    manifest = public_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    source = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    guide = render_guide(root, manifest, source)
    readme = render_readme(manifest, source)
    backup_dir.mkdir(parents=True, exist_ok=False)
    # Archive the complete internal manifest before moving any supporting files.
    shutil.copy2(manifest_path, backup_dir / "release_manifest.json")

    def install(path, write):
        fd, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary)
        try:
            write(temporary)
            if path.exists():
                backup = backup_dir / path.relative_to(root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
                shutil.copymode(path, temporary)
            else:
                os.chmod(temporary, 0o644)
            os.replace(temporary, path)
            LOG.info("Wrote public release file: %s", path.relative_to(root))
        finally:
            temporary.unlink(missing_ok=True)

    for name in ("source_identity_report", "negative_alias_report"):
        directory = root / name
        if directory.exists():
            shutil.move(str(directory), str(backup_dir / name))
            LOG.info("Retained internal report outside release: %s", backup_dir / name)
    install(manifest_path, lambda path: path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                       encoding="utf-8"))
    install(root / "GUIDE.md", lambda path: path.write_text(guide, encoding="utf-8"))
    install(root / "README.md", lambda path: path.write_text(readme, encoding="utf-8"))
    return manifest

