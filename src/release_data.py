"""Shared schemas and bounded-memory Parquet writing for dataset releases."""

import hashlib
import json
import logging
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq


LOG = logging.getLogger(__name__)
RELEASE_FORMAT_VERSION = 2
def retrieval_score_schema(schema):
    """The public format uses retrieval-score column names."""
    return schema


GENERATION_ORIGIN = pa.struct([
    ("source_query_id", pa.string()),
    ("query_type", pa.string()),
    ("difficulty", pa.string()),
    ("character", pa.string()),
    ("query_length_target", pa.int64()),
    ("metadata_json", pa.string()),
])
OTHER_QUERY = pa.struct([
    ("query_id", pa.int64()),
    ("mining_query_id", pa.int64()),
    ("query", pa.string()),
    ("query_word_count", pa.int64()),
    ("generation_metadata", pa.list_(GENERATION_ORIGIN)),
])
TRAIN_SCHEMA = pa.schema([
    ("query_id", pa.int64()),
    ("mining_query_id", pa.int64()),
    ("query", pa.string()),
    ("query_word_count", pa.int64()),
    ("generation_metadata", pa.list_(GENERATION_ORIGIN)),
    ("other_queries", pa.list_(OTHER_QUERY)),
    ("positive_idx", pa.int64()),
    ("negative_idxs", pa.list_(pa.int64())),
    ("negative_retrieval_scores", pa.list_(pa.float64())),
    ("negative_reranker_scores", pa.list_(pa.float64())),
    ("negative_reranker_raw_scores", pa.list_(pa.float64())),
    ("positive_reranker_score", pa.float64()),
    ("positive_reranker_raw_score", pa.float64()),
    ("skipped_idxs", pa.list_(pa.int64())),
    ("skipped_retrieval_scores", pa.list_(pa.float64())),
])
REF_SCHEMA = pa.schema([
    ("finepdf_idx", pa.int64()), ("finepdf_id", pa.string()),
    ("source_shard", pa.string()), ("row_in_shard", pa.int64()),
])
NEGATIVE_ALIAS_SCHEMA = pa.schema([
    ("query_id", pa.int64()), ("mining_query_id", pa.int64()),
    ("positive_mining_id", pa.int64()), ("positive_idx", pa.int64()),
    ("removed_mining_id", pa.int64()), ("canonical_idx", pa.int64()),
    ("retained_mining_id", pa.int64()), ("reason", pa.string()),
    ("retrieval_score", pa.float64()), ("reranker_score", pa.float64()),
    ("reranker_raw_score", pa.float64()),
])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    LOG.info("Wrote %s", path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ShardedWriter:
    """Write fixed-size shards without buffering an entire output shard."""

    def __init__(self, root, config, schema, rows_per_shard=100_000):
        self.root = Path(root)
        self.config = config
        self.directory = self.root / config
        self.directory.mkdir(parents=True)
        self.schema = schema
        self.capacity = rows_per_shard
        self.writer = None
        self.in_shard = 0
        self.rows = 0
        self.shards = []
        self.started = time.monotonic()
        self.last_report = self.started

    def write(self, table):
        table = table.cast(self.schema)
        offset = 0
        while offset < table.num_rows:
            if self.writer is None:
                self.path = self.directory / "part-{:05d}.parquet".format(len(self.shards))
                self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
                LOG.info("Opening output shard: %s", self.path)
            count = min(self.capacity - self.in_shard, table.num_rows - offset)
            self.writer.write_table(table.slice(offset, count), row_group_size=8192)
            self.rows += count
            self.in_shard += count
            offset += count
            if self.in_shard == self.capacity:
                self._close_shard()
        now = time.monotonic()
        if now - self.last_report >= 60:
            LOG.info("Writing %s: %s rows, %d completed shards, %.1f min elapsed",
                     self.config, format(self.rows, ",d"), len(self.shards), (now - self.started) / 60)
            self.last_report = now

    def _close_shard(self):
        if self.writer is not None:
            self.writer.close()
            LOG.info("Closed %s (%s rows); computing SHA256", self.path, format(self.in_shard, ",d"))
            self.shards.append({
                "path": self.path.relative_to(self.root).as_posix(),
                "start_row": self.rows - self.in_shard,
                "num_rows": self.in_shard,
                "sha256": sha256(self.path),
            })
            LOG.info("Output shard ready: %s (%.1f MiB; %s total %s rows)", self.path,
                     self.path.stat().st_size / (1024 * 1024), format(self.rows, ",d"), self.config)
            self.writer = None
            self.in_shard = 0

    def close(self):
        self._close_shard()
        LOG.info("Finished %s: %s rows in %d shards (%.1f s)", self.config,
                 format(self.rows, ",d"), len(self.shards), time.monotonic() - self.started)
        return {"num_rows": self.rows, "shards": self.shards}
