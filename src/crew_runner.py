"""
Orchestrates the Phase 1 crew: Requirements Agent -> SDLC Agent.

This is the "central orchestrator" for this two-agent slice: it builds the
LLM, wires up agents and tasks in sequence, runs the crew, times the run,
and returns structured results plus a plain-text audit record (in keeping
with the problem statement's emphasis on traceability/audit trails).
"""

import json
import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List

from crewai import Crew, Process

from .agents import build_llm, build_requirements_agent, build_sdlc_agent
from .analysis_engine import build_extended_analysis, mask_sensitive_data
from .document_loader import load_source_documents
from .models import RequirementsAnalysis, SDLCAnalysis
from .rag import RetrievedChunk, SourceDocument, retrieve_context
from .tasks import build_requirements_task, build_sdlc_task

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
AUDIT_ENCRYPTION_KEY = os.getenv("FINANCE_REQ_AUDIT_ENCRYPTION_KEY", "")


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
    return encrypted if os.path.isfile(encrypted) else plain


def _write_record(path: str, record: dict) -> None:
    data = json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8")
    if path.endswith(".enc"):
        from cryptography.fernet import Fernet
        data = Fernet(AUDIT_ENCRYPTION_KEY.encode("ascii")).encrypt(data)
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
        data = Fernet(AUDIT_ENCRYPTION_KEY.encode("ascii")).decrypt(data)
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

    llm = build_llm(model_name, base_url)

    requirements_agent = build_requirements_agent(llm)
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
    sdlc_task = build_sdlc_task(sdlc_agent, requirements_task)

    crew = Crew(
        agents=[requirements_agent, sdlc_agent],
        tasks=[requirements_task, sdlc_task],
        process=Process.sequential,
        verbose=True,
    )

    start = time.time()
    crew.kickoff()
    duration = time.time() - start

    requirements_output: RequirementsAnalysis = requirements_task.output.pydantic
    sdlc_output: SDLCAnalysis = sdlc_task.output.pydantic
    extended_analysis = build_extended_analysis(
        scenario=safe_scenario,
        requirements=requirements_output,
        sdlc=sdlc_output,
        retrieved_chunks=retrieved_chunks,
    )

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
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
    )
    _save_audit_record(result, actor)
    return result


def _save_audit_record(result: RunResult, actor: str) -> None:
    """Persist a JSON record of the run for traceability/audit purposes."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, f"run_{result.run_id}.json.enc" if AUDIT_ENCRYPTION_KEY else f"run_{result.run_id}.json")
    payload = {
        "run_id": result.run_id,
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


def save_approval_changes(run_id: str, approval_rows: list[dict], actor: str = "reviewer") -> str:
    """Persist reviewer decisions and append a hash-linked approval event."""
    approval_rows = _mask_metadata_value(approval_rows, "approval queue", [])
    path = _audit_path(run_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Run {run_id} was not found; approval changes were not saved.")
    record = _read_record(path)
    history = record.get("approval_history", [])
    event = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "actor": actor,
        "approvals": approval_rows,
        "previous_hash": history[-1]["event_hash"] if history else "GENESIS",
    }
    canonical = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    event["event_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    history.append(event)
    record["approval_history"] = history
    record["extended_analysis"]["human_approval_gates"] = approval_rows
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
    _write_record(path, record)
    return event["event_hash"]


def load_audit_record(run_id: str) -> dict:
    return _read_record(_audit_path(run_id))


def save_evaluation_feedback(run_id: str, feedback: dict, actor: str = "reviewer") -> str:
    feedback = _mask_metadata_value(feedback, "evaluation feedback", [])
    record = load_audit_record(run_id)
    history = record.get("approval_history", [])
    event = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "actor": actor,
        "type": "evaluation_feedback",
        "feedback": feedback,
        "previous_hash": history[-1]["event_hash"] if history else "GENESIS",
    }
    canonical = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    event["event_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    history.append(event)
    record["approval_history"] = history
    _write_record(_audit_path(run_id), record)
    return event["event_hash"]
