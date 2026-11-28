"""
MongoDB persistence layer for run history and audit records.

Stores completed agent runs, approval events, and reviewer feedback in MongoDB
so users can browse and reload past runs across sessions. When
FINANCE_REQ_AUDIT_ENCRYPTION_KEY is set, sensitive run payloads are Fernet-
encrypted before writing to MongoDB.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from pymongo import DESCENDING, MongoClient
from pymongo.errors import PyMongoError

_CLIENT_CACHE: dict[tuple[str, int], MongoClient] = {}
_INDEXED_TARGETS: set[tuple[str, str, str]] = set()
_RUN_ID_TS_RE = re.compile(r"^(\d{8})-(\d{6})")


def get_mongo_config() -> dict[str, Any]:
    """Return active MongoDB configuration from environment variables."""
    enabled = os.getenv("FINANCE_REQ_MONGODB_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
    return {
        "enabled": enabled,
        "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017").strip() or "mongodb://localhost:27017",
        "db_name": os.getenv("MONGODB_DB_NAME", "finance_req_agents").strip() or "finance_req_agents",
        "collection_name": os.getenv("MONGODB_COLLECTION", "runs").strip() or "runs",
    }


def reset_client_cache() -> None:
    """Close and clear cached MongoDB clients (primarily for tests)."""
    for client in _CLIENT_CACHE.values():
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    _CLIENT_CACHE.clear()
    _INDEXED_TARGETS.clear()


def _encryption_key() -> str:
    return os.getenv("FINANCE_REQ_AUDIT_ENCRYPTION_KEY", "").strip()


def _infer_created_at(run_id: str) -> str:
    match = _RUN_ID_TS_RE.match(run_id or "")
    if match:
        date_part, time_part = match.groups()
        try:
            parsed = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _get_client(uri: str, timeout_ms: int = 2000) -> MongoClient:
    key = (uri, timeout_ms)
    client = _CLIENT_CACHE.get(key)
    if client is None:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            socketTimeoutMS=5000,
        )
        _CLIENT_CACHE[key] = client
    return client


def get_collection(
    uri: str | None = None,
    db_name: str | None = None,
    collection_name: str | None = None,
    client: MongoClient | None = None,
    ensure_indexes: bool = True,
):
    """Return the configured MongoDB collection and ensure indexes exist."""
    cfg = get_mongo_config()
    if not cfg["enabled"] and client is None:
        raise RuntimeError("MongoDB integration is disabled via FINANCE_REQ_MONGODB_ENABLED=0.")
    target_uri = uri or cfg["uri"]
    target_db = db_name or cfg["db_name"]
    target_col = collection_name or cfg["collection_name"]

    mongo_client = client or _get_client(target_uri)
    collection = mongo_client[target_db][target_col]

    cache_key = (target_uri, target_db, target_col)
    if ensure_indexes and cache_key not in _INDEXED_TARGETS:
        collection.create_index("run_id", unique=True)
        collection.create_index([("created_at", DESCENDING)])
        collection.create_index("created_by")
        _INDEXED_TARGETS.add(cache_key)
    return collection


def check_mongo_connection(
    uri: str | None = None,
    db_name: str | None = None,
    collection_name: str | None = None,
    client: MongoClient | None = None,
) -> tuple[bool, str]:
    """Check whether MongoDB is reachable and return (connected, status_message)."""
    cfg = get_mongo_config()
    if not cfg["enabled"] and client is None:
        return False, "MongoDB disabled (FINANCE_REQ_MONGODB_ENABLED=0)."
    target_uri = uri or cfg["uri"]
    target_db = db_name or cfg["db_name"]
    target_col = collection_name or cfg["collection_name"]
    try:
        mongo_client = client or _get_client(target_uri)
        mongo_client.admin.command("ping")
        return True, f"{target_db}.{target_col}"
    except (PyMongoError, OSError, RuntimeError) as exc:
        return False, str(exc)


def _encode_document(record: dict) -> dict:
    """Prepare a run record for MongoDB storage, encrypting if configured."""
    run_id = str(record["run_id"])
    created_at = record.get("created_at") or _infer_created_at(run_id)
    updated_at = record.get("updated_at") or _utc_now_iso()
    created_by = str(record.get("created_by") or "local-unsecured-user")
    model_name = str(record.get("model_name") or "unknown")
    duration_seconds = float(record.get("duration_seconds") or 0.0)

    normalized = {
        **record,
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "created_by": created_by,
        "model_name": model_name,
        "duration_seconds": duration_seconds,
    }
    normalized.pop("_id", None)

    key = _encryption_key()
    if key:
        from cryptography.fernet import Fernet

        token = Fernet(key.encode("ascii")).encrypt(
            json.dumps(normalized, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        return {
            "run_id": run_id,
            "created_at": created_at,
            "updated_at": updated_at,
            "created_by": created_by,
            "model_name": model_name,
            "duration_seconds": duration_seconds,
            "encrypted": True,
            "encrypted_payload": token,
        }

    reqs = normalized.get("requirements") or {}
    sdlc = normalized.get("sdlc") or {}
    return {
        **normalized,
        "encrypted": False,
        "functional_count": len(reqs.get("functional_requirements", [])),
        "non_functional_count": len(reqs.get("non_functional_requirements", [])),
        "recommended_approach": sdlc.get("recommended_approach", ""),
        "approval_events_count": len(normalized.get("approval_history", [])),
    }


def _decode_document(doc: dict | None) -> dict | None:
    """Decode a MongoDB document back into a plain run record dict."""
    if not doc:
        return None
    cleaned = dict(doc)
    cleaned.pop("_id", None)

    if cleaned.get("encrypted"):
        key = _encryption_key()
        if not key:
            raise ValueError(
                f"Run {cleaned.get('run_id', '')} is encrypted in MongoDB, "
                "but FINANCE_REQ_AUDIT_ENCRYPTION_KEY is not configured."
            )
        from cryptography.fernet import Fernet

        plaintext = Fernet(key.encode("ascii")).decrypt(
            str(cleaned["encrypted_payload"]).encode("ascii")
        )
        decoded = json.loads(plaintext.decode("utf-8"))
        decoded.pop("_id", None)
        return decoded

    for meta_key in ("encrypted", "functional_count", "non_functional_count", "recommended_approach", "approval_events_count"):
        cleaned.pop(meta_key, None)
    return cleaned


def summarize_record(record: dict, source: str = "mongodb") -> dict:
    """Build a lightweight UI summary row from a full or encrypted run record."""
    run_id = str(record.get("run_id", ""))
    created_at = str(record.get("created_at") or _infer_created_at(run_id))
    updated_at = str(record.get("updated_at") or created_at)

    if record.get("encrypted") and "requirements" not in record:
        try:
            record = _decode_document(record) or record
        except Exception:  # noqa: BLE001
            return {
                "run_id": run_id,
                "created_at": created_at,
                "updated_at": updated_at,
                "created_by": record.get("created_by", "unknown"),
                "model_name": record.get("model_name", "unknown"),
                "duration_seconds": float(record.get("duration_seconds") or 0.0),
                "scenario": "[Encrypted record - key unavailable]",
                "scenario_preview": "[Encrypted record - key unavailable]",
                "functional_count": 0,
                "non_functional_count": 0,
                "recommended_approach": "Encrypted",
                "reference_documents": [],
                "uploaded_documents": [],
                "approval_events_count": 0,
                "source": source,
                "encrypted": True,
            }

    reqs = record.get("requirements") or {}
    sdlc = record.get("sdlc") or {}
    scenario = str(record.get("scenario_redacted") or record.get("scenario") or "").strip()
    preview = scenario if len(scenario) <= 140 else f"{scenario[:137]}..."
    return {
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "created_by": str(record.get("created_by") or "local-unsecured-user"),
        "model_name": str(record.get("model_name") or "unknown"),
        "duration_seconds": float(record.get("duration_seconds") or 0.0),
        "scenario": scenario,
        "scenario_preview": preview,
        "functional_count": len(reqs.get("functional_requirements", [])),
        "non_functional_count": len(reqs.get("non_functional_requirements", [])),
        "recommended_approach": str(sdlc.get("recommended_approach") or ""),
        "reference_documents": list(record.get("reference_documents") or []),
        "uploaded_documents": list(record.get("uploaded_documents") or []),
        "approval_events_count": len(record.get("approval_history") or []),
        "source": source,
        "encrypted": bool(_encryption_key()),
    }


def save_run_record(record: dict, *, collection=None) -> bool:
    """Insert or update a run record in MongoDB by run_id."""
    if not record or not record.get("run_id"):
        raise ValueError("Run record must include a non-empty 'run_id'.")
    target = collection if collection is not None else get_collection()
    doc = _encode_document(record)
    target.replace_one({"run_id": doc["run_id"]}, doc, upsert=True)
    return True


def get_run_record(run_id: str, *, collection=None) -> dict | None:
    """Fetch and decode a single run record from MongoDB by run_id."""
    target = collection if collection is not None else get_collection()
    doc = target.find_one({"run_id": run_id})
    return _decode_document(doc)


def list_run_summaries(
    limit: int = 100,
    search_query: str = "",
    model_filter: str = "",
    created_by: str | None = None,
    *,
    collection=None,
) -> list[dict]:
    """List recent runs stored in MongoDB, ordered newest first."""
    target = collection if collection is not None else get_collection()
    query: dict[str, Any] = {}
    if model_filter and model_filter != "All":
        query["model_name"] = model_filter
    if created_by:
        query["created_by"] = created_by

    projection = {
        "_id": 0,
        "run_id": 1,
        "created_at": 1,
        "updated_at": 1,
        "created_by": 1,
        "model_name": 1,
        "duration_seconds": 1,
        "scenario_redacted": 1,
        "scenario": 1,
        "reference_documents": 1,
        "uploaded_documents": 1,
        "requirements.functional_requirements": 1,
        "requirements.non_functional_requirements": 1,
        "sdlc.recommended_approach": 1,
        "approval_history": 1,
        "encrypted": 1,
        "encrypted_payload": 1,
    }
    cursor = target.find(query, projection).sort([("created_at", DESCENDING), ("run_id", DESCENDING)])
    if not search_query.strip():
        cursor = cursor.limit(max(1, limit))

    needle = search_query.strip().lower()
    summaries = []
    for doc in cursor:
        summary = summarize_record(doc, source="mongodb")
        if needle:
            haystack = " ".join([
                summary["run_id"],
                summary["model_name"],
                summary["created_by"],
                summary["scenario"],
                summary["recommended_approach"],
                " ".join(summary["reference_documents"]),
            ]).lower()
            if needle not in haystack:
                continue
        summaries.append(summary)
        if len(summaries) >= max(1, limit):
            break
    return summaries


def delete_run_record(run_id: str, *, collection=None) -> bool:
    """Delete a run record from MongoDB by run_id."""
    target = collection if collection is not None else get_collection()
    res = target.delete_one({"run_id": run_id})
    return res.deleted_count > 0


def count_run_records(*, collection=None) -> int:
    """Return the total number of stored run documents."""
    target = collection if collection is not None else get_collection()
    return int(target.count_documents({}))


def sync_outputs_dir_to_mongo(
    outputs_dir: str,
    read_record_fn=None,
    *,
    collection=None,
) -> dict[str, Any]:
    """Import existing JSON / encrypted JSON run files from outputs_dir into MongoDB.

    Existing records already present in MongoDB are skipped so approval/feedback
    updates in MongoDB are never overwritten by older local files.
    """
    if not os.path.isdir(outputs_dir):
        return {"synced": 0, "skipped": 0, "errors": []}

    target = collection if collection is not None else get_collection()
    existing_ids = {
        doc["run_id"]
        for doc in target.find({}, {"_id": 0, "run_id": 1})
        if doc.get("run_id")
    }

    synced = 0
    skipped = 0
    errors: list[str] = []
    key = _encryption_key()

    for filename in sorted(os.listdir(outputs_dir)):
        if not filename.startswith("run_"):
            continue
        if not (filename.endswith(".json") or filename.endswith(".json.enc")):
            continue
        path = os.path.join(outputs_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            if read_record_fn is not None:
                record = read_record_fn(path)
            else:
                with open(path, "rb") as stream:
                    raw = stream.read()
                if filename.endswith(".enc"):
                    if not key:
                        raise ValueError("FINANCE_REQ_AUDIT_ENCRYPTION_KEY is required to read .json.enc files.")
                    from cryptography.fernet import Fernet

                    raw = Fernet(key.encode("ascii")).decrypt(raw)
                record = json.loads(raw.decode("utf-8"))

            run_id = str(record.get("run_id") or "")
            if not run_id:
                run_id = filename.removeprefix("run_").removesuffix(".json.enc").removesuffix(".json")
                record["run_id"] = run_id

            if run_id in existing_ids:
                skipped += 1
                continue

            save_run_record(record, collection=target)
            existing_ids.add(run_id)
            synced += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{filename}: {exc}")

    return {"synced": synced, "skipped": skipped, "errors": errors}
