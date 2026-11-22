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
) -> RunResult:
    # Redact user-controlled text before either retrieval or LLM submission.
    safe_scenario, scenario_mask_events = mask_sensitive_data(scenario)
    safe_uploads = []
    source_mask_events = []
    for document in uploaded_documents or []:
        safe_text, events = mask_sensitive_data(document.text)
        safe_uploads.append(SourceDocument(name=document.name, text=safe_text, source_type=document.source_type))
        source_mask_events.extend({**event, "source": document.name} for event in events)

    llm = build_llm(model_name, base_url)

    requirements_agent = build_requirements_agent(llm)
    sdlc_agent = build_sdlc_agent(llm)

    source_documents = load_source_documents(reference_documents)
    for document in source_documents:
        document.text, events = mask_sensitive_data(document.text)
        source_mask_events.extend({**event, "source": document.name} for event in events)
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
        scenario=scenario,
        reference_documents=reference_documents,
        uploaded_documents=[doc.name for doc in uploaded_documents or []],
        retrieved_chunks=retrieved_chunks,
        extended_analysis=extended_analysis,
    )
    _save_audit_record(result)
    return result


def _save_audit_record(result: RunResult) -> None:
    """Persist a JSON record of the run for traceability/audit purposes."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, f"run_{result.run_id}.json")
    payload = {
        "run_id": result.run_id,
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_approval_changes(run_id: str, approval_rows: list[dict], actor: str = "reviewer") -> str:
    """Persist reviewer decisions and append a hash-linked approval event."""
    path = os.path.join(OUTPUTS_DIR, f"run_{run_id}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Run {run_id} was not found; approval changes were not saved.")
    with open(path, "r", encoding="utf-8") as stream:
        record = json.load(stream)
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
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
    return event["event_hash"]
