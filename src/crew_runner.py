"""
Orchestrates the Phase 1 crew: Requirements Agent -> SDLC Agent.

This is the "central orchestrator" for this two-agent slice: it builds the
LLM, wires up agents and tasks in sequence, runs the crew, times the run,
and returns structured results plus a plain-text audit record (in keeping
with the problem statement's emphasis on traceability/audit trails).
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List

from crewai import Crew, Process

from .agents import build_llm, build_requirements_agent, build_sdlc_agent
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


def run_crew(
    scenario: str,
    reference_documents: List[str],
    model_name: str,
    base_url: str = "http://localhost:11434",
    uploaded_documents: List[SourceDocument] | None = None,
    retrieval_top_k: int = 6,
) -> RunResult:
    llm = build_llm(model_name, base_url)

    requirements_agent = build_requirements_agent(llm)
    sdlc_agent = build_sdlc_agent(llm)

    source_documents = load_source_documents(reference_documents)
    if uploaded_documents:
        source_documents.extend(uploaded_documents)

    reference_text, retrieved_chunks = retrieve_context(
        query=scenario,
        documents=source_documents,
        top_k=retrieval_top_k,
    )
    requirements_task = build_requirements_task(requirements_agent, scenario, reference_text)
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

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
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
        "scenario": result.scenario,
        "reference_documents": result.reference_documents,
        "uploaded_documents": result.uploaded_documents,
        "retrieved_chunks": [chunk.__dict__ for chunk in result.retrieved_chunks],
        "requirements": result.requirements.model_dump(),
        "sdlc": result.sdlc.model_dump(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
