"""
Deterministic specialist agents/checkers used around the LLM crew.

The PDF problem statement calls for a broad multi-agent platform. Running a
large number of LLM agents locally would be slow and brittle for a classroom
prototype, so these helpers implement the repeatable parts as rule-based
specialists: clarification, classification, quality validation, compliance,
security/privacy, risk, documentation, traceability, SDLC scoring, and human
approval routing.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .models import RequirementItem, RequirementsAnalysis, SDLCAnalysis


SENSITIVE_PATTERNS = [
    (re.compile(r"\b\d{12}\b"), "[MASKED-AADHAAR-LIKE-ID]"),
    (re.compile(r"\b\d{16}\b"), "[MASKED-CARD-LIKE-ID]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE), "[MASKED-TAX-ID]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[MASKED-EMAIL]"),
    (re.compile(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b"), "[MASKED-PHONE]"),
]

DOMAIN_LABELS = {
    "Business": ("customer", "loan", "payment", "onboarding", "account", "policy", "business"),
    "Functional": ("shall", "submit", "approve", "reject", "route", "calculate", "generate"),
    "Security": ("auth", "mfa", "encrypt", "role", "access", "fraud", "security"),
    "Privacy": ("privacy", "personal", "mask", "retention", "consent", "pii"),
    "Regulatory": ("kyc", "aml", "regulation", "compliance", "audit", "legal"),
    "Performance": ("latency", "response", "throughput", "seconds", "concurrent"),
    "Availability": ("availability", "uptime", "recovery", "failover", "backup"),
    "Auditability": ("audit", "trace", "log", "evidence", "report"),
    "Integration": ("api", "legacy", "bureau", "core banking", "database"),
    "Operations": ("monitor", "incident", "support", "deployment", "rollback"),
}

REGULATION_RULES = [
    ("KYC/AML onboarding control", ("kyc", "aml", "identity", "customer onboarding", "sanction")),
    ("Financial audit and evidence retention", ("audit", "log", "evidence", "trace", "report")),
    ("Data privacy and retention control", ("privacy", "personal", "pii", "retention", "consent")),
    ("Access-control and least-privilege policy", ("role", "access", "permission", "mfa", "authentication")),
    ("Credit-risk policy", ("loan", "credit", "score", "underwriter", "eligibility")),
    ("Payment security control", ("payment", "transaction", "limit", "fraud", "settlement")),
]

SDLC_RULES = [
    ("Waterfall", ("stable requirements and extensive approvals", 50)),
    ("V-Model", ("strict verification, validation, and documentation", 58)),
    ("Spiral", ("high uncertainty or technical risk", 52)),
    ("Agile", ("frequently changing business requirements", 55)),
    ("DevSecOps", ("continuous secure delivery and automated controls", 60)),
    ("Agile-V-Model or DevSecOps hybrid", ("high regulation with evolving requirements", 68)),
]


def build_extended_analysis(
    scenario: str,
    requirements: RequirementsAnalysis,
    sdlc: SDLCAnalysis,
    retrieved_chunks: Iterable,
) -> dict:
    all_requirements = requirements.functional_requirements + requirements.non_functional_requirements
    retrieved_chunks = list(retrieved_chunks)
    masked_scenario, masking_events = mask_sensitive_data(scenario)
    enriched_requirements = [_enrich_requirement(req) for req in all_requirements]
    quality_issues = _quality_issues(enriched_requirements)
    compliance_matrix = _compliance_matrix(enriched_requirements)
    risk_register = _risk_register(enriched_requirements, quality_issues)
    traceability_matrix = _traceability_matrix(enriched_requirements, retrieved_chunks)
    decision_factors = _decision_factors(scenario, enriched_requirements, quality_issues)
    sdlc_scores = _score_sdlc(decision_factors, sdlc)

    return {
        "input_governance": {
            "masked_scenario": masked_scenario,
            "masking_events": masking_events,
            "accepted_sources": sorted({chunk.source_name for chunk in retrieved_chunks}),
            "source_allowlist_note": "Only selected knowledge-base files and explicit user uploads are retrieved.",
        },
        "stakeholder_interview_templates": _stakeholder_templates(),
        "clarification_questions": _clarification_questions(scenario, enriched_requirements),
        "validated_requirements": enriched_requirements,
        "quality_issues": quality_issues,
        "compliance_matrix": compliance_matrix,
        "security_privacy_controls": _security_privacy_controls(enriched_requirements),
        "risk_register": risk_register,
        "generated_artifacts": {
            "user_stories": _user_stories(enriched_requirements),
            "use_cases": _use_cases(enriched_requirements),
            "acceptance_criteria": _acceptance_catalog(enriched_requirements),
            "assumptions_and_dependencies": _assumptions_and_dependencies(requirements, enriched_requirements),
            "open_issues": _open_issues(quality_issues),
        },
        "traceability_matrix": traceability_matrix,
        "sdlc_decision_engine": {
            "decision_factors": decision_factors,
            "rule_scores": sdlc_scores,
            "workflow": _workflow(sdlc_scores[0]["approach"] if sdlc_scores else sdlc.recommended_approach),
        },
        "human_approval_gates": _approval_gates(enriched_requirements, sdlc_scores),
        "evaluation_metrics": _evaluation_metrics(enriched_requirements, quality_issues, compliance_matrix),
    }


def mask_sensitive_data(text: str) -> tuple[str, list[dict]]:
    masked = text
    events = []
    for pattern, token in SENSITIVE_PATTERNS:
        matches = pattern.findall(masked)
        if matches:
            masked = pattern.sub(token, masked)
            events.append({"token": token, "count": len(matches)})
    return masked, events


def _enrich_requirement(req: RequirementItem) -> dict:
    text = f"{req.category} {req.description} {req.reasoning}".lower()
    labels = [label for label, terms in DOMAIN_LABELS.items() if any(term in text for term in terms)]
    if not labels:
        labels = ["Business"]

    regulations = [name for name, terms in REGULATION_RULES if any(term in text for term in terms)]
    if not regulations and any(label in labels for label in ("Security", "Privacy", "Regulatory", "Auditability")):
        regulations = ["Organisation policy and compliance review"]

    risk_level = "High" if any(label in labels for label in ("Security", "Privacy", "Regulatory")) else req.risk_level
    confidence = req.confidence_score
    if "source" in req.reasoning.lower() or "chunk" in req.reasoning.lower() or "policy" in req.reasoning.lower():
        confidence = max(confidence, 82)
    if not regulations and "Non-functional" in req.id:
        confidence = min(confidence, 65)

    acceptance = req.acceptance_criteria or [
        f"Given valid inputs, when {req.description.removeprefix('The system shall ').rstrip('.')}, then the action is completed and recorded.",
        "The outcome is visible to authorised users and included in the audit trail.",
    ]

    return {
        **req.model_dump(),
        "multi_labels": labels,
        "applicable_regulations": req.applicable_regulations or regulations,
        "risk_level": risk_level,
        "confidence_score": confidence,
        "acceptance_criteria": acceptance,
        "approval_status": req.approval_status or "Pending human approval",
    }


def _quality_issues(requirements: list[dict]) -> list[dict]:
    issues = []
    seen = Counter(_normalise(req["description"]) for req in requirements)
    for req in requirements:
        text = req["description"].lower()
        checks = []
        if len(req["description"].split()) < 8:
            checks.append("too brief to be testable")
        if any(word in text for word in ("fast", "easy", "appropriate", "secure", "user-friendly")):
            checks.append("contains vague quality wording")
        if not req["reasoning"].strip():
            checks.append("missing source or business justification")
        if seen[_normalise(req["description"])] > 1:
            checks.append("possible duplicate")
        if not req["acceptance_criteria"]:
            checks.append("missing acceptance criteria")
        if checks:
            issues.append({
                "requirement_id": req["id"],
                "issue_type": ", ".join(checks),
                "recommended_action": "Return to the clarification agent before baseline approval.",
            })
    return issues


def _compliance_matrix(requirements: list[dict]) -> list[dict]:
    rows = []
    for req in requirements:
        for control in req["applicable_regulations"] or ["No explicit mapping - needs compliance review"]:
            rows.append({
                "requirement_id": req["id"],
                "control_or_policy": control,
                "evidence": req["reasoning"] or "Generated from stakeholder scenario",
                "gap": "Needs authorised compliance sign-off" if "review" in control.lower() or "no explicit" in control.lower() else "No obvious gap in prototype analysis",
                "approval_checkpoint": "Compliance/legal approval required before baseline",
            })
    return rows


def _security_privacy_controls(requirements: list[dict]) -> list[dict]:
    controls = [
        "Role-based access control for analyst, compliance, security, and approver roles",
        "MFA for privileged users and approval actions",
        "Encryption in transit and at rest for source documents and generated artefacts",
        "Prompt-injection resistant source framing and retrieval-source allowlisting",
        "Sensitive-data masking before LLM submission",
        "Session isolation and immutable audit logging",
    ]
    if any("Payment" in req["multi_labels"] for req in requirements):
        controls.append("Transaction-limit and step-up authentication checks")
    if any("Privacy" in req["multi_labels"] for req in requirements):
        controls.append("Retention, consent, and data-minimisation checks")
    return [{"control": control, "status": "Prototype control identified"} for control in controls]


def _risk_register(requirements: list[dict], quality_issues: list[dict]) -> list[dict]:
    risks = [
        {
            "risk_id": "RISK-001",
            "category": "Hallucination",
            "description": "LLM may produce unsupported regulatory or business claims.",
            "level": "High",
            "mitigation": "Use retrieved evidence, confidence scores, citations, and human approval.",
        },
        {
            "risk_id": "RISK-002",
            "category": "Prompt injection",
            "description": "Uploaded documents may contain instructions that conflict with the user's request.",
            "level": "High",
            "mitigation": "Treat retrieved sources as evidence only and use source allowlisting.",
        },
    ]
    if quality_issues:
        risks.append({
            "risk_id": "RISK-003",
            "category": "Requirement quality",
            "description": f"{len(quality_issues)} requirements need clarification before approval.",
            "level": "Medium",
            "mitigation": "Route flagged items to the clarification and validation agents.",
        })
    if any(req["risk_level"] == "High" for req in requirements):
        risks.append({
            "risk_id": "RISK-004",
            "category": "Security/compliance",
            "description": "High-impact financial requirements are present.",
            "level": "High",
            "mitigation": "Require security, compliance, and architecture sign-off.",
        })
    return risks


def _traceability_matrix(requirements: list[dict], chunks: list) -> list[dict]:
    default_source = chunks[0].source_name if chunks else "Stakeholder scenario"
    rows = []
    for req in requirements:
        rows.append({
            "requirement_id": req["id"],
            "source": default_source,
            "stakeholder": req["source_stakeholder"],
            "linked_controls": ", ".join(req["applicable_regulations"]) or "Pending mapping",
            "acceptance_criteria_count": len(req["acceptance_criteria"]),
            "approval_status": req["approval_status"],
        })
    return rows


def _decision_factors(scenario: str, requirements: list[dict], issues: list[dict]) -> list[dict]:
    text = " ".join([scenario] + [req["description"] for req in requirements]).lower()
    factors = [
        ("Regulatory criticality", "High" if any(k in text for k in ("kyc", "aml", "compliance", "audit", "loan", "payment")) else "Medium"),
        ("Security risk", "High" if any(k in text for k in ("fraud", "payment", "auth", "privacy", "personal")) else "Medium"),
        ("Requirement stability", "Medium" if issues else "High"),
        ("Legacy-system dependence", "High" if any(k in text for k in ("legacy", "core banking", "api")) else "Medium"),
        ("Need for continuous delivery", "High" if any(k in text for k in ("mobile", "digital", "online", "real-time")) else "Medium"),
        ("Testing and documentation burden", "High"),
        ("Human approval need", "High"),
    ]
    return [{"factor": factor, "value": value} for factor, value in factors]


def _score_sdlc(factors: list[dict], sdlc: SDLCAnalysis) -> list[dict]:
    values = {row["factor"]: row["value"] for row in factors}
    scores = []
    for approach, (reason, base) in SDLC_RULES:
        score = base
        if values.get("Regulatory criticality") == "High" and approach in ("V-Model", "Agile-V-Model or DevSecOps hybrid"):
            score += 18
        if values.get("Security risk") == "High" and approach in ("DevSecOps", "Agile-V-Model or DevSecOps hybrid"):
            score += 16
        if values.get("Requirement stability") != "High" and approach in ("Agile", "Spiral", "Agile-V-Model or DevSecOps hybrid"):
            score += 10
        if values.get("Need for continuous delivery") == "High" and approach in ("Agile", "DevSecOps", "Agile-V-Model or DevSecOps hybrid"):
            score += 10
        if sdlc.recommended_approach.lower() in approach.lower() or approach.lower() in sdlc.recommended_approach.lower():
            score += 5
        scores.append({"approach": approach, "score_percent": min(score, 96), "rule_basis": reason})
    return sorted(scores, key=lambda item: item["score_percent"], reverse=True)


def _workflow(top_approach: str) -> list[dict]:
    return [
        {"phase": "Intake and discovery", "roles": "Business analyst, product owner, stakeholders", "deliverables": "Interview notes, source inventory, masked input pack", "gate": "Stakeholder review"},
        {"phase": "Requirement extraction and classification", "roles": "Requirement, classification, and clarification agents", "deliverables": "Structured requirements, questions, issue list", "gate": "Requirement quality gate"},
        {"phase": "Security, privacy, compliance, and risk analysis", "roles": "Security, compliance, risk, architecture", "deliverables": "Control matrix, risk register, threat notes", "gate": "Security/compliance approval"},
        {"phase": f"{top_approach} delivery planning", "roles": "Project manager, architect, QA lead", "deliverables": "SDLC workflow, roles, entry/exit criteria", "gate": "SDLC sign-off"},
        {"phase": "Implementation and verification", "roles": "Developers, QA, security team", "deliverables": "Code, tests, SAST/DAST evidence, traceability updates", "gate": "Release readiness review"},
        {"phase": "Deployment, monitoring, and improvement", "roles": "Operations, audit, compliance", "deliverables": "Deployment approval, monitoring records, change log", "gate": "Production approval"},
    ]


def _stakeholder_templates() -> list[dict]:
    return [
        {"role": "Customer/end user", "questions": "Goals, pain points, expected workflow, usability needs"},
        {"role": "Business/product owner", "questions": "Business objectives, rules, priorities, exceptions, success metrics"},
        {"role": "Compliance/legal officer", "questions": "Jurisdiction, regulations, evidence, retention, approval checkpoints"},
        {"role": "Security team", "questions": "Authentication, authorisation, encryption, threat model, incident response"},
        {"role": "Architect/developer", "questions": "Legacy integrations, APIs, data model, constraints, deployment model"},
        {"role": "Operations/audit", "questions": "Monitoring, reporting, audit logs, support, recovery objectives"},
    ]


def _clarification_questions(scenario: str, requirements: list[dict]) -> list[str]:
    text = scenario.lower()
    questions = []
    if not any(word in text for word in ("role", "customer", "analyst", "admin", "underwriter")):
        questions.append("Which user roles must use or approve this workflow?")
    if not any(word in text for word in ("regulation", "kyc", "aml", "audit", "compliance")):
        questions.append("Which jurisdiction, policies, and compliance controls apply?")
    if not any(word in text for word in ("latency", "seconds", "availability", "uptime", "rto", "rpo")):
        questions.append("What performance, availability, RTO, and RPO targets are required?")
    if not any(req["applicable_regulations"] for req in requirements):
        questions.append("Which approved knowledge-base sources should be used to map controls?")
    if not questions:
        questions.append("Are the generated requirements ready to baseline, or should any be modified?")
    return questions


def _user_stories(requirements: list[dict]) -> list[dict]:
    stories = []
    for req in requirements:
        if req["id"].startswith("FR"):
            stories.append({
                "requirement_id": req["id"],
                "story": f"As an authorised financial-services user, I want {req['description'].removeprefix('The system shall ').rstrip('.')}, so that the business process is completed with evidence.",
                "priority": req["priority"],
            })
    return stories


def _use_cases(requirements: list[dict]) -> list[dict]:
    return [
        {
            "use_case_id": f"UC-{idx:02d}",
            "linked_requirement": req["id"],
            "primary_actor": "Authorised user",
            "main_success_scenario": req["description"],
            "alternate_or_exception_flow": "If required data is missing or a control fails, route to manual review.",
        }
        for idx, req in enumerate(requirements, start=1)
        if req["id"].startswith("FR")
    ]


def _acceptance_catalog(requirements: list[dict]) -> list[dict]:
    return [
        {"requirement_id": req["id"], "criteria": criterion}
        for req in requirements
        for criterion in req["acceptance_criteria"]
    ]


def _assumptions_and_dependencies(requirements: RequirementsAnalysis, enriched: list[dict]) -> list[dict]:
    rows = [{"type": "Assumption", "item": item} for item in requirements.assumptions]
    for req in enriched:
        for dependency in req["dependencies"]:
            rows.append({"type": "Dependency", "item": f"{req['id']}: {dependency}"})
    if not rows:
        rows.append({"type": "Assumption", "item": "Detailed jurisdiction, SLA, and approval-role data will be confirmed by stakeholders."})
    return rows


def _open_issues(quality_issues: list[dict]) -> list[str]:
    if not quality_issues:
        return ["Confirm final baseline approval and source evidence with authorised stakeholders."]
    return [f"{issue['requirement_id']}: {issue['issue_type']}" for issue in quality_issues]


def _approval_gates(requirements: list[dict], scores: list[dict]) -> list[dict]:
    gates = [
        {"item": "Final requirement baseline", "required_approver": "Product owner and business analyst", "status": "Pending"},
        {"item": "Regulatory interpretations", "required_approver": "Compliance/legal officer", "status": "Pending"},
        {"item": "High-risk security requirements", "required_approver": "Security lead", "status": "Pending"},
        {"item": "Architecture-critical requirements", "required_approver": "Software architect", "status": "Pending"},
        {"item": f"SDLC selection: {scores[0]['approach'] if scores else 'Pending'}", "required_approver": "Project manager, architect, security, compliance", "status": "Pending"},
        {"item": "Production readiness", "required_approver": "Operations, audit, compliance", "status": "Pending"},
    ]
    for req in requirements:
        if req["risk_level"] == "High":
            gates.append({"item": req["id"], "required_approver": "Security/compliance owner", "status": req["approval_status"]})
    return gates


def _evaluation_metrics(requirements: list[dict], issues: list[dict], compliance_rows: list[dict]) -> list[dict]:
    total = max(1, len(requirements))
    mapped = sum(1 for req in requirements if req["applicable_regulations"])
    cited = sum(1 for req in requirements if req["reasoning"].strip())
    return [
        {"metric": "Requirement completeness proxy", "value": f"{100 - round((len(issues) / total) * 100)}%"},
        {"metric": "Traceability coverage", "value": f"{round((cited / total) * 100)}%"},
        {"metric": "Regulatory/control mapping coverage", "value": f"{round((mapped / total) * 100)}%"},
        {"metric": "Compliance matrix rows", "value": str(len(compliance_rows))},
        {"metric": "Human correction queue", "value": str(len(issues))},
        {"metric": "Hallucination control", "value": "RAG evidence + source-only framing + approval gates"},
    ]


def _normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()
