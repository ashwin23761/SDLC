"""Orchestrates the requirements, clarification, validation, and SDLC agents."""

import json
import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from pymongo.errors import PyMongoError

from .analysis_engine import build_extended_analysis, mask_sensitive_data
from .db import (
    check_mongo_connection,
    delete_run_record,
    get_run_record,
    list_run_summaries,
    save_run_record,
    summarize_record,
    sync_outputs_dir_to_mongo,
)
from .document_loader import load_source_documents
from .models import (
    ClarificationAnalysis,
    RequirementsAnalysis,
    RequirementsValidation,
    SDLCAnalysis,
)
from .rag import RetrievedChunk, SourceDocument, retrieve_context

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
AUDIT_ENCRYPTION_KEY = os.getenv("FINANCE_REQ_AUDIT_ENCRYPTION_KEY", "")


def _audit_encryption_key() -> str:
    return os.getenv("FINANCE_REQ_AUDIT_ENCRYPTION_KEY", AUDIT_ENCRYPTION_KEY).strip()


def _mask_metadata_value(value, source_name: str, events: list) :
    if isinstance(value, str):
        safe, found = mask_sensitive_data(value)
        events.extend({**event, "source": source_name, "field": "metadata"} for event in found)
        return safe
    if isinstance(value, list):
        return [_mask_metadata_value(item, source_name, events) for item in value]
    if isinstance(value, dict):
        return {key: _mask_metadata_value(item, source_name, events) for key, item in value.items()}
    return value


def _audit_path(run_id: str) -> str:
    encrypted = os.path.join(OUTPUTS_DIR, f"run_{run_id}.json.enc")
    plain = os.path.join(OUTPUTS_DIR, f"run_{run_id}.json")
    if os.path.isfile(encrypted):
        return encrypted
    if os.path.isfile(plain):
        return plain
    return encrypted if _audit_encryption_key() else plain


def _write_record(path: str, record: dict) -> None:
    data = json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8")
    if path.endswith(".enc"):
        from cryptography.fernet import Fernet
        data = Fernet(_audit_encryption_key().encode("ascii")).encrypt(data)
    temporary_path = path + ".tmp"
    with open(temporary_path, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def _read_record(path: str) -> dict:
    with open(path, "rb") as stream:
        data = stream.read()
    if path.endswith(".enc"):
        from cryptography.fernet import Fernet
        data = Fernet(_audit_encryption_key().encode("ascii")).decrypt(data)
    return json.loads(data.decode("utf-8"))


@dataclass
class RunResult:
    run_id: str
    model_name: str
    duration_seconds: float
    requirements: RequirementsAnalysis
    sdlc: SDLCAnalysis
    scenario: str
    reference_documents: List[str]
    uploaded_documents: List[str]
    retrieved_chunks: List[RetrievedChunk]
    extended_analysis: dict
    created_by: str = "local-unsecured-user"
    created_at: str = ""
    storage_info: dict = field(default_factory=dict)


def run_crew(
    scenario: str,
    reference_documents: List[str],
    model_name: str,
    base_url: str = "http://localhost:11434",
    uploaded_documents: List[SourceDocument] | None = None,
    retrieval_top_k: int = 6,
    actor: str = "local-unsecured-user",
) -> RunResult:
    # Redact user-controlled text before either retrieval or LLM submission.
    safe_scenario, scenario_mask_events = mask_sensitive_data(scenario)
    safe_uploads = []
    source_mask_events = []
    for document in uploaded_documents or []:
        safe_text, events = mask_sensitive_data(document.text)
        safe_metadata = _mask_metadata_value(document.metadata, document.name, source_mask_events)
        safe_uploads.append(SourceDocument(name=document.name, text=safe_text, source_type=document.source_type, metadata=safe_metadata))
        source_mask_events.extend({**event, "source": document.name} for event in events)

    from crewai import Crew, Process

    from .agents import (
        build_clarification_agent,
        build_llm,
        build_requirements_agent,
        build_sdlc_agent,
        build_validation_agent,
    )
    from .tasks import (
        build_clarification_task,
        build_requirements_task,
        build_sdlc_task,
        build_validation_task,
    )

    llm = build_llm(model_name, base_url)

    requirements_agent = build_requirements_agent(llm)
    clarification_agent = build_clarification_agent(llm)
    validation_agent = build_validation_agent(llm)
    sdlc_agent = build_sdlc_agent(llm)

    source_documents = load_source_documents(reference_documents)
    for document in source_documents:
        document.text, events = mask_sensitive_data(document.text)
        source_mask_events.extend({**event, "source": document.name} for event in events)
        document.metadata = _mask_metadata_value(document.metadata, document.name, source_mask_events)
    source_documents.extend(safe_uploads)

    reference_text, retrieved_chunks = retrieve_context(
        query=safe_scenario,
        documents=source_documents,
        top_k=retrieval_top_k,
    )
    requirements_task = build_requirements_task(requirements_agent, safe_scenario, reference_text)
    clarification_task = build_clarification_task(
        clarification_agent,
        requirements_task,
        safe_scenario,
        reference_text,
    )
    validation_task = build_validation_task(
        validation_agent,
        requirements_task,
        clarification_task,
        reference_text,
    )
    sdlc_task = build_sdlc_task(sdlc_agent, requirements_task, validation_task)

    crew = Crew(
        agents=[
            requirements_agent,
            clarification_agent,
            validation_agent,
            sdlc_agent,
        ],
        tasks=[
            requirements_task,
            clarification_task,
            validation_task,
            sdlc_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    start = time.time()
    crew.kickoff()
    duration = time.time() - start

    requirements_output: RequirementsAnalysis = requirements_task.output.pydantic
    clarification_output: ClarificationAnalysis = clarification_task.output.pydantic
    validation_output: RequirementsValidation = validation_task.output.pydantic
    sdlc_output: SDLCAnalysis = sdlc_task.output.pydantic
    extended_analysis = build_extended_analysis(
        scenario=safe_scenario,
        requirements=requirements_output,
        sdlc=sdlc_output,
        retrieved_chunks=retrieved_chunks,
    )
    extended_analysis["clarification_agent"] = clarification_output.model_dump()
    extended_analysis["validation_agent"] = validation_output.model_dump()

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    extended_analysis["input_governance"]["masking_events"] = scenario_mask_events + source_mask_events
    result = RunResult(
        run_id=run_id,
        model_name=model_name,
        duration_seconds=duration,
        requirements=requirements_output,
        sdlc=sdlc_output,
        scenario=safe_scenario,
        reference_documents=reference_documents,
        uploaded_documents=[doc.name for doc in uploaded_documents or []],
        retrieved_chunks=retrieved_chunks,
        extended_analysis=extended_analysis,
        created_by=actor,
        created_at=created_at,
    )
    result.storage_info = _save_audit_record(result, actor)
    return result


def _try_save_mongo(record: dict) -> tuple[bool, str]:
    try:
        save_run_record(record)
        return True, ""
    except (PyMongoError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("MongoDB persistence skipped for run %s: %s", record.get("run_id"), exc)
        return False, str(exc)


def _save_audit_record(result: RunResult, actor: str) -> dict:
    """Persist a run record to local audit storage and MongoDB."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, f"run_{result.run_id}.json.enc" if _audit_encryption_key() else f"run_{result.run_id}.json")
    now_iso = result.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = {
        "run_id": result.run_id,
        "created_at": now_iso,
        "updated_at": now_iso,
        "created_by": actor,
        "model_name": result.model_name,
        "duration_seconds": result.duration_seconds,
        "scenario_redacted": result.scenario,
        "security_note": "Sensitive patterns were masked before retrieval and LLM submission.",
        "reference_documents": result.reference_documents,
        "uploaded_documents": result.uploaded_documents,
        "retrieved_chunks": [chunk.__dict__ for chunk in result.retrieved_chunks],
        "requirements": result.requirements.model_dump(),
        "sdlc": result.sdlc.model_dump(),
        "extended_analysis": result.extended_analysis,
    }
    _write_record(path, payload)
    mongo_saved, mongo_error = _try_save_mongo(payload)
    return {"file_path": path, "mongodb_saved": mongo_saved, "mongodb_error": mongo_error}


def save_approval_changes(run_id: str, approval_rows: list[dict], actor: str = "reviewer") -> str:
    """Persist reviewer decisions and append a hash-linked approval event."""
    approval_rows = _mask_metadata_value(approval_rows, "approval queue", [])
    record = load_audit_record(run_id)
    history = record.get("approval_history", [])
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    event = {
        "timestamp": now_iso,
        "actor": actor,
        "approvals": approval_rows,
        "previous_hash": history[-1]["event_hash"] if history else "GENESIS",
    }
    canonical = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    event["event_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    history.append(event)
    record["approval_history"] = history
    record["updated_at"] = now_iso
    extended = record.setdefault("extended_analysis", {})
    extended["human_approval_gates"] = approval_rows
    modifications = record.get("human_modified_requirements", {})
    for row in approval_rows:
        replacement = str(row.get("modified_output", "")).strip()
        if replacement:
            modifications[str(row.get("item", "Unspecified item"))] = {
                "replacement": replacement,
                "reviewer_notes": row.get("reviewer_notes", ""),
                "modified_by": actor,
                "event_hash": event["event_hash"],
            }
    record["human_modified_requirements"] = modifications
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    _write_record(_audit_path(run_id), record)
    _try_save_mongo(record)
    return event["event_hash"]


def load_audit_record(run_id: str) -> dict:
    try:
        mongo_record = get_run_record(run_id)
        if mongo_record is not None:
            return mongo_record
    except (PyMongoError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("Falling back to file storage while loading run %s: %s", run_id, exc)

    path = _audit_path(run_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Run {run_id} was not found in MongoDB or local outputs/.")
    return _read_record(path)


def save_evaluation_feedback(run_id: str, feedback: dict, actor: str = "reviewer") -> str:
    feedback = _mask_metadata_value(feedback, "evaluation feedback", [])
    record = load_audit_record(run_id)
    history = record.get("approval_history", [])
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    event = {
        "timestamp": now_iso,
        "actor": actor,
        "type": "evaluation_feedback",
        "feedback": feedback,
        "previous_hash": history[-1]["event_hash"] if history else "GENESIS",
    }
    canonical = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    event["event_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    history.append(event)
    record["approval_history"] = history
    record["updated_at"] = now_iso
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    _write_record(_audit_path(run_id), record)
    _try_save_mongo(record)
    return event["event_hash"]


def record_to_run_result(record: dict) -> RunResult:
    """Reconstruct a RunResult object from a stored MongoDB or JSON run record."""
    scenario = str(record.get("scenario_redacted") or record.get("scenario") or "")
    requirements = RequirementsAnalysis(**(record.get("requirements") or {"summary": ""}))
    sdlc = SDLCAnalysis(
        **(
            record.get("sdlc")
            or {
                "recommended_approach": "Unspecified",
                "workflow_notes": "No SDLC workflow notes recorded.",
            }
        )
    )
    retrieved_chunks = [
        RetrievedChunk(
            source_name=chunk.get("source_name", "unknown"),
            source_type=chunk.get("source_type", "knowledge-base"),
            chunk_index=int(chunk.get("chunk_index", 1)),
            score=float(chunk.get("score", 0.0)),
            text=chunk.get("text", ""),
            metadata=dict(chunk.get("metadata") or {}),
        )
        for chunk in record.get("retrieved_chunks", [])
    ]
    extended = record.get("extended_analysis")
    if not extended:
        extended = build_extended_analysis(
            scenario=scenario,
            requirements=requirements,
            sdlc=sdlc,
            retrieved_chunks=retrieved_chunks,
        )
    return RunResult(
        run_id=str(record["run_id"]),
        model_name=str(record.get("model_name") or "unknown"),
        duration_seconds=float(record.get("duration_seconds") or 0.0),
        requirements=requirements,
        sdlc=sdlc,
        scenario=scenario,
        reference_documents=list(record.get("reference_documents") or []),
        uploaded_documents=list(record.get("uploaded_documents") or []),
        retrieved_chunks=retrieved_chunks,
        extended_analysis=extended,
        created_by=str(record.get("created_by") or "local-unsecured-user"),
        created_at=str(record.get("created_at") or ""),
        storage_info={"loaded_from_history": True},
    )


def load_run_result(run_id: str) -> RunResult:
    """Load a run record by ID and return a UI-ready RunResult."""
    return record_to_run_result(load_audit_record(run_id))


def sync_local_history_to_mongo() -> dict:
    """Import any local outputs/ run files not yet present in MongoDB."""
    return sync_outputs_dir_to_mongo(OUTPUTS_DIR, read_record_fn=_read_record)


def list_run_history(
    limit: int = 100,
    search_query: str = "",
    model_filter: str = "",
    created_by: str | None = None,
    auto_sync_local: bool = True,
) -> tuple[list[dict], str, dict]:
    """List past run summaries from MongoDB, falling back to local outputs/ files.

    Returns (summaries, backend_name, sync_stats).
    """
    connected, _ = check_mongo_connection()
    if connected:
        sync_stats = {"synced": 0, "skipped": 0, "errors": []}
        try:
            if auto_sync_local:
                sync_stats = sync_local_history_to_mongo()
            summaries = list_run_summaries(
                limit=limit,
                search_query=search_query,
                model_filter=model_filter,
                created_by=created_by,
            )
            return summaries, "mongodb", sync_stats
        except (PyMongoError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("MongoDB listing failed, falling back to local outputs/: %s", exc)

    summaries = []
    needle = search_query.strip().lower()
    if os.path.isdir(OUTPUTS_DIR):
        for filename in sorted(os.listdir(OUTPUTS_DIR), reverse=True):
            if not filename.startswith("run_") or not (filename.endswith(".json") or filename.endswith(".json.enc")):
                continue
            path = os.path.join(OUTPUTS_DIR, filename)
            if not os.path.isfile(path):
                continue
            try:
                record = _read_record(path)
                summary = summarize_record(record, source="local_file")
            except Exception:  # noqa: BLE001
                continue
            if model_filter and model_filter != "All" and summary["model_name"] != model_filter:
                continue
            if created_by and summary["created_by"] != created_by:
                continue
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
    summaries.sort(key=lambda item: (item["created_at"], item["run_id"]), reverse=True)
    return summaries[: max(1, limit)], "local_files", {"synced": 0, "skipped": 0, "errors": []}


def delete_run_history_item(run_id: str) -> dict[str, bool]:
    """Delete a run from MongoDB and local outputs/ if present."""
    mongo_deleted = False
    try:
        mongo_deleted = delete_run_record(run_id)
    except (PyMongoError, OSError, RuntimeError) as exc:
        logger.warning("Failed deleting run %s from MongoDB: %s", run_id, exc)

    file_deleted = False
    for suffix in (".json", ".json.enc"):
        candidate = os.path.join(OUTPUTS_DIR, f"run_{run_id}{suffix}")
        if os.path.isfile(candidate):
            os.remove(candidate)
            file_deleted = True
    return {"mongodb": mongo_deleted, "file": file_deleted}
