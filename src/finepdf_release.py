"""Resolve pinned FinePDFs row references and emit contrastive text examples.

Select config="eng_Latn" or config="fra_Latn" at the multilingual root.
This packaged helper also accepts a standalone language directory. Downloads are lazy; exporting the
annotation release never downloads or publishes source text.
"""

from bisect import bisect_right
from collections import OrderedDict
import json
import operator
from pathlib import Path

import pyarrow.parquet as pq

from release_data import retrieval_score_schema


def language_release_dir(release_dir, config=None):
    """Select a language from the combined release, or use a language directory.

    Integer IDs are language-local. Requiring config at a multilingual root
    prevents accidentally resolving one language's indices against another.
    """
    root = Path(release_dir)
    path = root / "release_manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {}
    languages = manifest.get("language_configs")
    if languages is None:
        if config is not None:
            source = json.loads((root / "source_manifest.json").read_text())
            if source.get("config") != config:
                raise ValueError("Requested config does not match this language directory")
        return root
    if not (root / "_SUCCESS").is_file():
        raise ValueError("Combined release lacks _SUCCESS")
    if config is None:
        raise ValueError("Choose a language config: {}".format(", ".join(sorted(languages))))
    if config not in languages:
        raise ValueError("Unknown language config {!r}; choose {}".format(config, ", ".join(sorted(languages))))
    relative = Path(languages[config]["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid language directory in combined release manifest")
    selected = root / relative
    if root.resolve() not in selected.resolve().parents:
        raise ValueError("Language directory escapes the combined release")
    if not (selected / "_SUCCESS").is_file():
        raise ValueError("Language release lacks _SUCCESS: {}".format(config))
    source = json.loads((selected / "source_manifest.json").read_text())
    if source.get("config") != config:
        raise ValueError("Language directory and source manifest disagree")
    return selected


def read_source_manifest(release_dir, config=None):
    root = language_release_dir(release_dir, config=config)
    source = json.loads((root / "source_manifest.json").read_text())
    offset = 0
    seen = set()
    for shard in source["shards"]:
        path = Path(shard["path"])
        if path.is_absolute() or ".." in path.parts or shard["path"] in seen:
            raise ValueError("Invalid or duplicate source shard path")
        if shard["start_row"] != offset or shard["num_rows"] < 0:
            raise ValueError("Source manifest has inconsistent row offsets")
        seen.add(shard["path"])
        offset += shard["num_rows"]
    if offset != source["num_rows"] or not source["shards"]:
        raise ValueError("Source manifest row count mismatch")
    return source


class FinePDFResolver:
    """Cache source Parquet files and a bounded number of text row groups.

    local_source_root, when given, must contain the same pinned snapshot used
    during export. Otherwise hf_hub_download fetches the exact declared commit.
    """

    def __init__(self, release_dir, local_source_root=None, cache_dir=None,
                 max_cached_row_groups=4, max_open_shards=8, config=None):
        if max_cached_row_groups <= 0 or max_open_shards <= 0:
            raise ValueError("Cache limits must be positive")
        self.source = read_source_manifest(release_dir, config=config)
        self.local_root = Path(local_source_root) if local_source_root is not None else None
        self.cache_dir = cache_dir
        self.max_groups = max_cached_row_groups
        self.max_shards = max_open_shards
        self.shards = self.source["shards"]
        self.starts = [s["start_row"] for s in self.shards]
        self.files = OrderedDict()
        self.groups = OrderedDict()

    def _locate(self, index):
        index = operator.index(index)
        if index < 0 or index >= self.source["num_rows"]:
            raise IndexError("FinePDFs index outside the pinned corpus: {}".format(index))
        shard_index = bisect_right(self.starts, index) - 1
        return shard_index, index - self.starts[shard_index]

    def _file(self, shard_index):
        if shard_index in self.files:
            self.files.move_to_end(shard_index)
            return self.files[shard_index]
        shard = self.shards[shard_index]
        if self.local_root is not None:
            path = self.local_root / shard["path"]
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=self.source["repo_id"], repo_type="dataset",
                                   filename=shard["path"], revision=self.source["revision"],
                                   cache_dir=self.cache_dir)
        pf = pq.ParquetFile(path)
        if pf.metadata.num_rows != shard["num_rows"]:
            pf.close()
            raise ValueError("Source shard row count differs from the pinned manifest: {}".format(path))
        offsets, offset = [], 0
        for group in range(pf.metadata.num_row_groups):
            offsets.append(offset)
            offset += pf.metadata.row_group(group).num_rows
        self.files[shard_index] = (pf, offsets)
        if len(self.files) > self.max_shards:
            _, (old, _) = self.files.popitem(last=False)
            old.close()
        return pf, offsets

    def get_documents(self, indices):
        """Read each required row group once per call, preserving request order."""
        requests = {}
        indices = list(indices)
        for position, index in enumerate(indices):
            shard, row = self._locate(index)
            requests.setdefault(shard, []).append((position, row))
        results = [None] * len(indices)
        for shard, shard_requests in requests.items():
            pf, offsets = self._file(shard)
            group_requests = {}
            for position, row in shard_requests:
                group = bisect_right(offsets, row) - 1
                group_requests.setdefault(group, []).append((position, row - offsets[group]))
            for group, items in group_requests.items():
                key = (shard, group)
                if key not in self.groups:
                    self.groups[key] = pf.read_row_group(group, columns=["id", "text"])
                    if len(self.groups) > self.max_groups:
                        self.groups.popitem(last=False)
                self.groups.move_to_end(key)
                table = self.groups[key]
                for position, row in items:
                    results[position] = {"id": table["id"][row].as_py(),
                                         "text": table["text"][row].as_py()}
                    if results[position]["text"] is None:
                        raise ValueError("Source document has null text")
        return results

    def get_texts(self, indices):
        return [r["text"] for r in self.get_documents(indices)]

    def __getitem__(self, index):
        try:
            scalar = operator.index(index)
        except TypeError:
            return self.get_texts(index)
        return self.get_texts([scalar])[0]

    def close(self):
        for pf, _ in self.files.values():
            pf.close()
        self.files.clear()
        self.groups.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_source_dataset(release_dir, local_source_root=None, config=None, **kwargs):
    """Load ALL source text with the exact file order defining public row IDs.

    This is a regular, locally cached Dataset, which can require substantial
    disk space. Use FinePDFResolver for selective shard access instead.
    """
    from datasets import concatenate_datasets, load_dataset
    if kwargs.get("streaming"):
        raise ValueError("Source row indices require a regular Dataset; streaming has no random access")
    source = read_source_manifest(release_dir, config=config)
    # Explicit per-file concatenation avoids relying on a loader's glob sorting.
    from urllib.parse import quote
    datasets = []
    for shard in source["shards"]:
        if local_source_root is not None:
            path = str(Path(local_source_root) / shard["path"])
        else:
            path = "https://huggingface.co/datasets/{}/resolve/{}/{}".format(
                source["repo_id"], source["revision"], quote(shard["path"], safe="/"))
        dataset = load_dataset("parquet", data_files={source["split"]: [path]},
                               split=source["split"], **kwargs)
        if len(dataset) != shard["num_rows"]:
            raise ValueError("Loaded source shard row count differs from manifest")
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def iter_release_rows(release_dir, config=None):
    """Iterate local train shards in the manifest's explicit order."""
    root = language_release_dir(release_dir, config=config)
    if not (root / "_SUCCESS").is_file():
        raise ValueError("Release lacks _SUCCESS; export may be incomplete")
    manifest = json.loads((root / "release_manifest.json").read_text())
    if manifest["format_version"] not in (1, 2):
        raise ValueError("Unsupported release format version")
    for shard in manifest["configs"]["train"]["shards"]:
        with pq.ParquetFile(root / shard["path"]) as pf:
            if pf.metadata.num_rows != shard["num_rows"]:
                raise ValueError("Training shard row count differs from release manifest")
            schema = retrieval_score_schema(pf.schema_arrow)
            for batch in pf.iter_batches(batch_size=256):
                yield from batch.rename_columns(schema.names).to_pylist()


def iter_training_examples(rows, resolver, num_negatives=8, all_queries=False):
    """Yield text-only n-tuples in retrieval-score order, with no teacher-score reuse.

    All selected negatives are resolved in one bulk request per grouped row.
    Fail rather than pad when there are fewer negatives than requested.
    """
    if num_negatives < 0:
        raise ValueError("num_negatives must be nonnegative")
    for row in rows:
        if len(row["negative_idxs"]) < num_negatives:
            raise ValueError("Query {} has only {} negatives; requested {}".format(
                row["query_id"], len(row["negative_idxs"]), num_negatives))
        texts = resolver.get_texts([row["positive_idx"]] + row["negative_idxs"][:num_negatives])
        examples = [row] + (row["other_queries"] if all_queries else [])
        for query in examples:
            yield {"query": query["query"], "positive": texts[0],
                   **{"negative_{}".format(i): text for i, text in enumerate(texts[1:])}}

