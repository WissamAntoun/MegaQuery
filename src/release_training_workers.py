"""Bounded, ordered CPU workers for release training-row conversion."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from release_data import TRAIN_SCHEMA, NEGATIVE_ALIAS_SCHEMA


_REMAP = None
_POLICY = None
_EXCLUDED = ()
COUNTERS = ("removed_self_aliases", "removed_duplicate_aliases", "removed_empty_negatives",
            "removed_empty_skipped", "affected_training_rows", "empty_negative_rows",
            "candidate_count_before", "candidate_count_after")
REASONS = {"alias_of_positive": "removed_self_aliases",
           "duplicate_negative_alias": "removed_duplicate_aliases",
           "empty_missing_negative": "removed_empty_negatives",
           "empty_missing_skipped": "removed_empty_skipped"}


class QueryBatches:
    """Take the next N metadata rows without decoding them in the parent."""

    def __init__(self, path, batch_size):
        self.source = pq.ParquetFile(path)
        self.reader = self.source.iter_batches(batch_size=batch_size, use_threads=False)
        self.batch = None
        self.offset = 0

    def take(self, count):
        if count <= 0:
            raise ValueError("Training batch has no query metadata")
        slices = []
        while count:
            if self.batch is None or self.offset == self.batch.num_rows:
                self.batch = next(self.reader, None)
                self.offset = 0
                if self.batch is None:
                    raise ValueError("Query metadata ended before the mining rows")
            size = min(count, self.batch.num_rows - self.offset)
            slices.append(self.batch.slice(self.offset, size))
            self.offset += size
            count -= size
        return pa.Table.from_batches(slices)

    def finish(self):
        if ((self.batch is not None and self.offset < self.batch.num_rows)
                or next(self.reader, None) is not None):
            raise ValueError("Unconsumed query metadata")

    def close(self):
        self.source.close()


def convert_batch(job, remap, alias_policy, excluded_document_ids=()):
    """Keep the existing per-row validation and alias selection unchanged."""
    from export_finepdf_release import training_row
    batch, metadata, next_query_id = job
    queries = metadata.to_pylist()
    rows, removed, counts = [], [], set()
    counters = dict.fromkeys(COUNTERS, 0)
    offset = 0
    for record in batch.to_pylist():
        size = len(record["query_texts"])
        group = queries[offset:offset + size]
        if not group or any((q["positive_idx"], q["doc_id_int"]) !=
                            (record["positive_idx"], record["doc_id_int"]) for q in group):
            raise ValueError("Query metadata and mining rows are not aligned")
        start = len(removed)
        row = training_row(record, group, remap, next_query_id, alias_policy, removed, excluded_document_ids)
        counters["affected_training_rows"] += len(removed) != start
        counters["empty_negative_rows"] += not row["negative_idxs"]
        counters["candidate_count_before"] += len(record["neg_embed"])
        counters["candidate_count_after"] += len(row["negative_idxs"])
        next_query_id += size
        offset += size
        counts.add(len(row["negative_idxs"]))
        rows.append(row)
    if offset != len(queries):
        raise ValueError("Query count differs from the mining batch")
    for removal in removed:
        counters[REASONS[removal["reason"]]] += 1
    return {"train": pa.Table.from_pylist(rows, schema=TRAIN_SCHEMA),
            "audit": pa.Table.from_pylist(removed, schema=NEGATIVE_ALIAS_SCHEMA) if removed else None,
            "counters": counters, "next_query_id": next_query_id, "negative_counts": counts}


def initialize_worker(map_path, map_size, alias_policy, excluded_document_ids):
    global _REMAP, _POLICY, _EXCLUDED
    pa.set_cpu_count(1)
    # np.asarray is a zero-copy view: the mapping stays backed by the same file,
    # while scalar access avoids np.memmap's Python __getitem__ wrapper.
    _REMAP = np.asarray(np.memmap(map_path, mode="r", dtype="int64", shape=(map_size,)))
    _POLICY = alias_policy
    _EXCLUDED = excluded_document_ids


def worker_batch(job):
    return convert_batch(job, _REMAP, _POLICY, _EXCLUDED)


def ordered_results(executor, jobs, max_pending):
    """Submit a fixed window; never queue the entire corpus like Pool.map."""
    jobs = iter(jobs)
    pending = deque()
    for _ in range(max_pending):
        job = next(jobs, None)
        if job is None:
            break
        pending.append(executor.submit(worker_batch, job))
    while pending:
        yield pending.popleft().result()
        job = next(jobs, None)
        if job is not None:
            pending.append(executor.submit(worker_batch, job))


def training_batches(reader, query_path, remap, batch_size, workers, alias_policy, excluded_document_ids=()):
    """Read Arrow batches centrally; decode/validate/convert them in workers."""
    metadata = QueryBatches(query_path, batch_size)

    def jobs():
        next_query_id = 0
        for batch in reader:
            size = pc.sum(pc.list_value_length(batch.column(batch.schema.get_field_index("qids")))).as_py()
            if size is None:
                raise ValueError("Mining batch has missing query IDs")
            yield batch, metadata.take(size), next_query_id
            next_query_id += size
        metadata.finish()

    try:
        if workers == 1:
            view = np.asarray(remap)
            for job in jobs():
                yield convert_batch(job, view, alias_policy, excluded_document_ids)
        else:
            # Spawn avoids inheriting a running DuckDB connection and its native
            # threads. Each worker opens only the read-only integer map.
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialize_worker,
                                     initargs=(str(remap.filename), len(remap), alias_policy, excluded_document_ids)) as pool:
                yield from ordered_results(pool, jobs(), max_pending=2 * workers)
    finally:
        metadata.close()
