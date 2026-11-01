"""Decode generation query strings and recover exact missing mining keys."""

import ast
import itertools
import json
import logging
import pickletools
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


LOG = logging.getLogger(__name__)


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def parquet_source(paths):
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    return "read_parquet([{}], union_by_name=true)".format(
        ",".join(literal(path) for path in paths))


def decoded_pickled_string(value, validate_unicode=True):
    """Read a primitive pickle string by inspecting opcodes; never unpickle objects."""
    if not value.startswith(b"\x80") or len(value) > 1_000_000:
        return None
    try:
        operations = []
        for operation in pickletools.genops(value):
            operations.append(operation)
            if len(operations) > 5:
                return None
        if not operations or operations[0][0].name != "PROTO" or operations[0][1] not in (2, 3, 4, 5):
            return None
        if operations[-1][0].name != "STOP" or operations[-1][2] != len(value) - 1:
            return None
        body = operations[1:-1]
        if body and body[0][0].name == "FRAME":
            _, size, position = body.pop(0)
            if size != len(value) - position - 9:
                return None
        if len(body) not in (1, 2) or body[0][0].name not in {"BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}:
            return None
        if len(body) == 2 and body[1][0].name not in {"MEMOIZE", "BINPUT", "LONG_BINPUT"}:
            return None
        text = body[0][1]
        if validate_unicode:
            text.encode("utf-8", errors="strict")
        return text.strip()
    except (ValueError, UnicodeError, OverflowError):
        return None


def serialized_query_literal(value):
    """Recognize a stored byte representation of a primitive serialized string."""
    if not isinstance(value, str) or not value.startswith(("b'\\x80", 'b"\\x80')) or len(value) > 1_000_000:
        return False
    try:
        data = ast.literal_eval(value)
        return isinstance(data, bytes) and decoded_pickled_string(data, validate_unicode=False) is not None
    except (ValueError, SyntaxError, RecursionError):
        return False


def query_text(value):
    """Decode Arrow binary queries as UTF-8 or a verified primitive pickle string."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError:
            decoded = decoded_pickled_string(value)
            if decoded is None:
                raise
            return decoded
    return str(value).strip()


def decoded_bytes_literal(value):
    if not value.startswith(("b'", 'b"')) or len(value) > 1_000_000:
        return None
    try:
        decoded = ast.literal_eval(value)
        return query_text(decoded) if isinstance(decoded, bytes) else None
    except (ValueError, SyntaxError, UnicodeError, RecursionError):
        return None


def recover_query_keys(connection, generation_path, work, batch_size=1024):
    """Add metadata aliases; retain original keys and all historical field values.

    An alias is accepted only for an unmatched, exact (UUID, mined query) pair.
    Existing matches take precedence, including legitimate literal-looking text.
    Multiple encoding variants that decode to one key merge their origins.
    """
    work = Path(work)
    connection.execute("CREATE OR REPLACE TEMP VIEW encoding_generation AS SELECT * FROM {}".format(
        parquet_source(generation_path)))
    connection.execute("CREATE OR REPLACE TEMP VIEW encoded_generation_queries AS "
                       "SELECT * FROM encoding_generation WHERE left(query,2) IN ('b''','b\"')")
    candidates = connection.execute("SELECT count(*) FROM encoded_generation_queries").fetchone()[0]
    summary = {"python_bytes_literal_candidates": candidates, "recovered_metadata_pairs": 0,
               "policy": "UTF-8 or primitive pickle-string decoding accepted only for an exact missing UUID/mined-query pair; existing matches win."}
    if not candidates:
        return [generation_path], summary
    LOG.info("Checking %s byte-literal generation queries against exact missing mining keys", format(candidates, ",d"))
    connection.execute("CREATE OR REPLACE TEMP TABLE encoding_missing_keys AS "
                       "SELECT DISTINCT d.finepdf_id,q.query FROM "
                       "(SELECT doc_id_int,unnest(query_texts) AS query FROM mined) q "
                       "JOIN doc_mapping d ON q.doc_id_int=d.old_idx "
                       "ANTI JOIN encoding_generation g ON d.finepdf_id=g.finepdf_id AND q.query=g.query")
    if not connection.execute("SELECT count(*) FROM encoding_missing_keys").fetchone()[0]:
        return [generation_path], summary
    schema = pa.schema([("finepdf_id", pa.string()), ("query", pa.string()),
                        ("generation_metadata", pa.string())])
    candidates_path = work / "generation_utf8_candidates.parquet"
    reader = connection.execute("SELECT e.* FROM encoded_generation_queries e "
                                "SEMI JOIN (SELECT DISTINCT finepdf_id FROM encoding_missing_keys) m USING(finepdf_id)").fetch_record_batch(batch_size)
    decoded_count = 0
    with pq.ParquetWriter(candidates_path, schema, compression="zstd") as writer:
        for batch in reader:
            records = []
            for record in batch.to_pylist():
                decoded = decoded_bytes_literal(record["query"])
                if decoded is None:
                    continue
                origins = json.loads(record["generation_metadata"])
                for origin in origins:
                    origin["query_encoding_recovery"] = {
                        "format": "python_bytes_literal_pickle_string" if ast.literal_eval(record["query"]).startswith(b"\x80")
                        else "python_bytes_literal_utf8", "original_query": record["query"]}
                records.append({"finepdf_id": record["finepdf_id"], "query": decoded,
                                "generation_metadata": json.dumps(origins, ensure_ascii=False)})
            if records:
                writer.write_table(pa.Table.from_pylist(records, schema=schema))
                decoded_count += len(records)
    summary["decoded_candidate_rows"] = decoded_count
    reader = connection.execute("SELECT c.* FROM {} c SEMI JOIN encoding_missing_keys m "
                                "USING(finepdf_id,query) ORDER BY c.finepdf_id,c.query".format(
                                    parquet_source(candidates_path))).fetch_record_batch(batch_size)
    rows = (record for batch in reader for record in batch.to_pylist())
    aliases_path = work / "generation_utf8_aliases.parquet"
    buffer, recovered = [], 0
    with pq.ParquetWriter(aliases_path, schema, compression="zstd") as writer:
        for (identifier, query), group in itertools.groupby(rows, lambda row: (row["finepdf_id"], row["query"])):
            origins = [origin for record in group for origin in json.loads(record["generation_metadata"])]
            buffer.append({"finepdf_id": identifier, "query": query,
                           "generation_metadata": json.dumps(origins, ensure_ascii=False)})
            recovered += 1
            if len(buffer) >= batch_size:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    summary["recovered_metadata_pairs"] = recovered
    LOG.info("Recovered exact metadata keys for %s document/query pairs: %s", format(recovered, ",d"), aliases_path)
    return [generation_path, aliases_path], summary
