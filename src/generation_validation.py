#!/usr/bin/env python3
"""Check release generation Parquets without scanning the mining corpus."""

import glob
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq


LOG = logging.getLogger(__name__)


def open_generation(path):
    try:
        pf = pq.ParquetFile(path)
        if not {"id", "query"} <= set(pf.schema_arrow.names):
            pf.close()
            raise ValueError("Generation input must contain id and query")
        return pf
    except Exception as exc:
        raise ValueError("Cannot read generation Parquet {}: {}".format(path, exc)) from exc


def generation_batches(path, pf, columns, batch_size):
    try:
        yield from pf.iter_batches(batch_size=batch_size, columns=columns, use_threads=False)
    except Exception as exc:
        raise ValueError("Cannot decode generation Parquet {}: {}".format(path, exc)) from exc


def check_generation(paths, read_data=False, batch_size=1024):
    """Inspect every file, collecting failures instead of hiding the last path."""
    records = []
    for number, path in enumerate(paths, 1):
        record = {"path": str(path)}
        try:
            record["size_bytes"] = path.stat().st_size
            with open_generation(path) as pf:
                record["num_rows"] = pf.metadata.num_rows
                if read_data:
                    # Validate generation data, including metadata, but avoid
                    # decoding the potentially large retained document text.
                    columns = [name for name in pf.schema_arrow.names
                               if name not in {"text", "pos_text", "doc_text"}]
                    for _ in generation_batches(path, pf, columns, batch_size):
                        pass
            record["status"] = "ok"
            LOG.info("Generation check %d/%d: %s (%s rows)",
                     number, len(paths), path, format(record["num_rows"], ",d"))
        except Exception as exc:
            record.update(status="invalid", error=str(exc))
            LOG.error("Generation check %d/%d failed: %s", number, len(paths), exc)
        records.append(record)
    return {"check": "decode" if read_data else "footer_and_schema",
            "files": records, "invalid_files": sum(r["status"] != "ok" for r in records)}


def validate_generation(paths):
    report = check_generation(paths)
    failures = [r for r in report["files"] if r["status"] != "ok"]
    if failures:
        raise ValueError("Invalid generation inputs ({} files):\n{}\n"
                         "Restore these files, select the correct generation inputs, or explicitly "
                         "exclude them with --exclude-generation-file. Metadata coverage for every "
                         "mined query remains required.".format(
                             len(failures), "\n".join(r["error"] for r in failures)))
    return report

