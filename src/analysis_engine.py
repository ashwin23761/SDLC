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
import os
from collections import Counter
from typing import Iterable

from .models import RequirementItem, RequirementsAnalysis, SDLCAnalysis


SENSITIVE_PATTERNS = [
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[MASKED-AADHAAR-LIKE-ID]"),
    (re.compile(r"\b(?:\d[ -]?){15}\d\b"), "[MASKED-CARD-LIKE-ID]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE), "[MASKED-TAX-ID]"),
    (re.compile(r"\b(?:account|acct)(?:\s*(?:number|no\.?|#))?\s*[:=-]?\s*\d{8,18}\b", re.IGNORECASE), "[MASKED-ACCOUNT-ID]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.IGNORECASE), "[MASKED-IBAN-LIKE-ID]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[MASKED-EMAIL]"),
    (re.compile(r"\b(?:\+\d{1,3}[- ]?)?[6-9]\d{2}[- ]?\d{3}[- ]?\d{4}\b"), "[MASKED-PHONE]"),
]

DOMAIN_LABELS = {
    "Business": ("customer", "loan", "payment", "onboarding", "account", "policy", "business"),
    "Stakeholder": ("customer", "end user", "business analyst", "product owner", "underwriter", "operator"),
    "Functional": ("shall", "submit", "approve", "reject", "route", "calculate", "generate"),
    "Security": ("authentication", "authorisation", "authorization", "mfa", "encrypt", "role", "access", "fraud", "security"),
    "Privacy": ("privacy", "personal", "mask", "retention", "consent", "pii"),
    "Regulatory": ("kyc", "aml", "regulation", "compliance", "audit", "legal"),
    "Performance": ("latency", "response", "throughput", "seconds", "concurrent"),
    "Availability": ("availability", "uptime", "recovery", "failover", "backup"),
    "Reliability": ("reliability", "resilience", "recovery", "failover", "backup"),
    "Usability": ("usability", "accessible", "user-friendly", "workflow", "screen"),
    "Data management": ("data", "retention", "consent", "record", "database"),
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
    decision_factors = _decision_factors(scenario, enriched_requirements, quality_issues)
    sdlc_scores = _score_sdlc(decision_factors, sdlc)

    traceability_matrix = _traceability_matrix(enriched_requirements, retrieved_chunks)
    return {
        "input_governance": {
            "masked_scenario": masked_scenario,
            "masking_events": masking_events,
            "accepted_sources": sorted({chunk.source_name for chunk in retrieved_chunks}),
            "source_metadata": {chunk.source_name: chunk.metadata for chunk in retrieved_chunks},
            "source_allowlist_note": "Only selected knowledge-base files and explicit user uploads are retrieved.",
        },
        "stakeholder_interview_templates": _stakeholder_templates(),
        "clarification_questions": _clarification_questions(scenario, enriched_requirements),
        "validated_requirements": enriched_requirements,
        "quality_issues": quality_issues,
        "compliance_matrix": compliance_matrix,
        "fairness_review": _fairness_review(enriched_requirements),
        "regulatory_change_controls": _regulatory_change_controls(retrieved_chunks),
        "security_privacy_controls": _security_privacy_controls(enriched_requirements),
        "risk_register": risk_register,
        "generated_artifacts": {
            "srs": _srs_document(requirements, enriched_requirements),
            "user_stories": _user_stories(enriched_requirements),
            "use_cases": _use_cases(enriched_requirements),
            "acceptance_criteria": _acceptance_catalog(enriched_requirements),
            "assumptions_and_dependencies": _assumptions_and_dependencies(requirements, enriched_requirements),
            "open_issues": _open_issues(quality_issues),
            "process_workflows": _process_workflows(enriched_requirements),
            "data_requirements": _artefact_requirements(enriched_requirements, ("data", "privacy", "retention", "consent", "customer", "identity")),
            "interface_requirements": _artefact_requirements(enriched_requirements, ("api", "integration", "bureau", "service", "interface", "legacy")),
            "threat_register": _threat_register(enriched_requirements),
        },
        "traceability_matrix": traceability_matrix,
        "sdlc_decision_engine": {
            "decision_factors": decision_factors,
            "rule_scores": sdlc_scores,
            "workflow": _workflow(sdlc_scores[0]["approach"] if sdlc_scores else sdlc.recommended_approach),
        },
        "human_approval_gates": _approval_gates(enriched_requirements, sdlc_scores),
        "evaluation_metrics": _evaluation_metrics(enriched_requirements, quality_issues, compliance_matrix, traceability_matrix),
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
        regulations = ["Applicable jurisdiction and organisation policy: confirm with compliance/legal"]

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
        "source_references": req.source_references,
    }


def _quality_issues(requirements: list[dict]) -> list[dict]:
    issues = []
    seen = Counter(_normalise(req["description"]) for req in requirements)
    normalized = [(req, _normalise(req["description"])) for req in requirements]
    for index, req in enumerate(requirements):
        text = req["description"].lower()
        checks = []
        if len(req["description"].split()) < 8:
            checks.append("too brief to be testable")
        if any(word in text for word in ("fast", "easy", "appropriate", "secure", "user-friendly", "quickly", "sufficient", "as needed", "etc.")):
            checks.append("contains vague quality wording")
        if not req["reasoning"].strip():
            checks.append("missing source or business justification")
        if seen[_normalise(req["description"])] > 1:
            checks.append("possible duplicate")
        if not req["acceptance_criteria"]:
            checks.append("missing acceptance criteria")
        if not req.get("source_references") and not req["reasoning"].strip():
            checks.append("missing source traceability")
        if re.search(r"\b(TBD|TBC|appropriate|reasonable|as soon as possible|etc\.)\b", text):
            checks.append("contains undefined or unresolved terminology")
        if re.search(r"\b(must be feasible|best|optimal|modern|user-friendly)\b", text):
            checks.append("possible infeasible or subjective statement")
        if req["confidence_score"] < 60:
            checks.append("low confidence; human clarification required")
        if req["risk_level"] in ("High", "Critical") and not req["applicable_regulations"]:
            checks.append("high-impact requirement is missing an explicit control mapping")
        # Similarity-based duplicate/conflict heuristic; flagged pairs require human review.
        words = set(_normalise(req["description"]).split())
        for other_index, (other, other_norm) in enumerate(normalized):
            if other_index <= index:
                continue
            other_words = set(other_norm.split())
            union = words | other_words
            similarity = len(words & other_words) / len(union) if union else 0
            if similarity >= 0.82 and _normalise(req["description"]) != other_norm:
                checks.append(f"possible near-duplicate or conflict with {other['id']}")
            shared = words & other_words
            opposites = (("allow", "deny"), ("approve", "reject"), ("retain", "delete"))
            phrase_conflict = (
                "auto approve" in _normalise(req["description"]) and "manual review" in other_norm
            ) or (
                "manual review" in _normalise(req["description"]) and "auto approve" in other_norm
            )
            if (len(shared) >= 3 and any(a in words and b in other_words or b in words and a in other_words for a, b in opposites)) or phrase_conflict:
                checks.append(f"possible conflicting requirement with {other['id']}")
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
                "jurisdiction": "Unspecified - stakeholder confirmation required",
                "mapping_basis": "Heuristic candidate mapping; not a legal determination",
                "gap": "Needs authorised compliance sign-off",
                "approval_checkpoint": "Compliance/legal approval required before baseline",
            })
    return rows


def _fairness_review(requirements: list[dict]) -> list[dict]:
    decision_reqs = [req for req in requirements if any(term in f"{req['category']} {req['description']}".lower() for term in ("approve", "decline", "reject", "credit score", "eligibility", "risk rating", "automated decision"))]
    if not decision_reqs:
        return [{"status": "Not triggered by detected decisioning requirements", "note": "Human review is still needed for relevant financial decisions."}]
    checks = [
        "Confirm lawful and documented use of each decision feature and any proxy variables.",
        "Assess approval, decline, error, and override outcomes across relevant customer groups where lawful.",
        "Provide understandable adverse-action reasons and an accessible human review or appeal path.",
        "Review training/evaluation data for representativeness and validate changes before release.",
    ]
    return [{"requirement_id": req["id"], "decision_statement": req["description"], "fairness_checks": checks, "status": "Mandatory domain-expert review; no automated fairness determination"} for req in decision_reqs]


def _regulatory_change_controls(chunks: list) -> list[dict]:
    rows = []
    for source, metadata in sorted({chunk.source_name: chunk.metadata for chunk in chunks}.items()):
        rows.append({
            "source": source,
            "jurisdiction": metadata.get("jurisdiction", "Unknown"),
            "effective_date": metadata.get("effective_date", "Unknown"),
            "version": metadata.get("version", "Unknown"),
            "owner": metadata.get("owner", "Unknown"),
            "review_status": metadata.get("review_status", "Unknown"),
            "action": "Assign an owner, verify currency and applicability, record changes, and rerun impact analysis before relying on this source.",
        })
    if not rows:
        rows.append({"source": "No retrieved policy source", "action": "Select approved, versioned sources and confirm their jurisdiction and effective date."})
    return rows


def _security_privacy_controls(requirements: list[dict]) -> list[dict]:
    secure_mode = os.getenv("FINANCE_REQ_AUTH_REQUIRED", "0").strip().lower() in {"1", "true", "yes"}
    encrypted = bool(os.getenv("FINANCE_REQ_AUDIT_ENCRYPTION_KEY"))
    controls = [
        ("Role-based access control for analyst, compliance, security, and approver roles", "Enabled by configured local authentication" if secure_mode else "Disabled in local prototype mode"),
        ("MFA for privileged users and approval actions", "Password plus TOTP enabled in secure mode" if secure_mode else "Disabled in local prototype mode"),
        ("Encryption in transit and at rest for source documents and generated artefacts", "Audit encryption enabled" if encrypted else "Run records are plaintext; configure audit encryption"),
        ("Prompt-injection resistant source framing and retrieval-source allowlisting", "Source framing and explicit-source allowlist implemented; not a complete injection guarantee"),
        ("Sensitive-data masking before LLM submission", "Implemented for configured identifier patterns; not comprehensive PII detection"),
        ("Session isolation and immutable audit logging", "Session-specific UI state; approval history is hash-linked but files remain locally editable"),
        ("Retention and secure deletion", "Not implemented; configure deployment storage and retention policy"),
        ("Model and knowledge-base versioning", "Source metadata capture available; model/version governance not enforced"),
    ]
    if any("Payment" in req["multi_labels"] for req in requirements):
        controls.append(("Transaction-limit and step-up authentication checks", "Candidate requirement control; runtime enforcement not implemented"))
    if any("Privacy" in req["multi_labels"] for req in requirements):
        controls.append(("Retention, consent, and data-minimisation checks", "Candidate requirement control; runtime enforcement not implemented"))
    return [{"control": control, "status": status} for control, status in controls]


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
    if any(any(term in req["description"].lower() for term in ("credit score", "eligibility", "automated decision", "approve", "decline")) for req in requirements):
        risks.append({
            "risk_id": "RISK-005",
            "category": "Fairness and bias",
            "description": "Automated financial decisions may create disparate outcomes or use unjustified proxy variables.",
            "level": "High",
            "mitigation": "Require lawful-feature review, subgroup outcome analysis, explainable adverse action, and a human appeal path.",
        })
    return risks


def _traceability_matrix(requirements: list[dict], chunks: list) -> list[dict]:
    rows = []
    for req in requirements:
        citations = [str(value) for value in req.get("source_references", [])]
        cited_chunks = [chunk for chunk in chunks if any(ref.lower() in chunk.source_name.lower() for ref in citations)]
        if not cited_chunks:
            terms = set(_normalise(f"{req['category']} {req['description']}").split())
            ranked = []
            for chunk in chunks:
                chunk_terms = set(_normalise(chunk.text).split())
                overlap = len(terms & chunk_terms) / max(1, len(terms))
                if overlap >= 0.2:
                    ranked.append((overlap, chunk))
            cited_chunks = [chunk for _, chunk in sorted(ranked, key=lambda item: item[0], reverse=True)[:3]]
        sources = sorted({f"{chunk.source_name} (chunk {chunk.chunk_index})" for chunk in cited_chunks})
        if citations:
            sources = sorted(set(sources + citations))
        rows.append({
            "requirement_id": req["id"],
            "source": "; ".join(sources) if sources else "Stakeholder scenario (source-specific citation required)",
            "stakeholder": req["source_stakeholder"],
            "linked_controls": ", ".join(req["applicable_regulations"]) or "Pending mapping",
            "acceptance_criteria_count": len(req["acceptance_criteria"]),
            "confidence_score": req["confidence_score"],
            "evidence_status": "Model-cited source; verify" if citations else "Lexical candidate evidence; verify" if cited_chunks else "No retrieved source match",
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
        ("Project size", "Large" if any(k in text for k in ("enterprise", "multi-country", "platform", "millions of")) else "Medium"),
        ("Expected change frequency", "High" if any(k in text for k in ("frequent change", "evolving", "rapidly changing", "agile")) else "Unspecified"),
        ("Stakeholder availability", "Unspecified - confirm during discovery"),
        ("Budget and schedule constraints", "Unspecified - confirm during discovery"),
        ("Formal verification need", "High" if any(k in text for k in ("regulatory", "safety-critical", "audit", "compliance")) else "Medium"),
        ("Failure consequence", "High" if any(k in text for k in ("loan", "payment", "fraud", "credit", "customer data")) else "Unspecified"),
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
        if values.get("Formal verification need") == "High" and approach in ("V-Model", "Agile-V-Model or DevSecOps hybrid"):
            score += 12
        if values.get("Failure consequence") == "High" and approach in ("V-Model", "Spiral", "Agile-V-Model or DevSecOps hybrid"):
            score += 8
        if values.get("Project size") == "Large" and approach in ("Spiral", "DevSecOps", "Agile-V-Model or DevSecOps hybrid"):
            score += 6
        if values.get("Expected change frequency") == "High" and approach in ("Agile", "Spiral", "Agile-V-Model or DevSecOps hybrid"):
            score += 8
        if values.get("Legacy-system dependence") == "High" and approach in ("Spiral", "DevSecOps", "Agile-V-Model or DevSecOps hybrid"):
            score += 5
        if values.get("Stakeholder availability", "").lower().startswith("low") and approach in ("Agile", "Spiral"):
            score -= 6
        if sdlc.recommended_approach.lower() in approach.lower() or approach.lower() in sdlc.recommended_approach.lower():
            score += 3
        scores.append({"approach": approach, "score_percent": min(score, 96), "rule_basis": reason})
    return sorted(scores, key=lambda item: item["score_percent"], reverse=True)


def _workflow(top_approach: str) -> list[dict]:
    return [
        {"phase": "Intake and discovery", "activities": "Interview stakeholders; inventory sources; record jurisdiction, scope, budget, schedule, and constraints", "roles": "Business analyst, product owner, stakeholders", "deliverables": "Interview records, source inventory, redacted input pack", "security_controls": "Data minimisation and masked identifiers", "entry_criteria": "Sponsor and use case identified", "exit_criteria": "Scope and unknowns reviewed by stakeholders", "validation_gates": "Stakeholder review", "compliance_checkpoints": "Jurisdiction and applicability confirmed by compliance"},
        {"phase": "Requirement extraction and classification", "activities": "Extract, classify, deduplicate, link evidence, and ask follow-ups for missing inputs", "roles": "Requirements analyst, classification and clarification specialists", "deliverables": "SRS draft, questions, issue list, traceability links", "security_controls": "Source allowlist and prompt-injection isolation", "entry_criteria": "Approved sources and scenario available", "exit_criteria": "Every requirement has source, owner, priority, criteria, confidence", "validation_gates": "Completeness, consistency, and testability review", "compliance_checkpoints": "Candidate control mappings identified"},
        {"phase": "Security, privacy, compliance, and risk analysis", "activities": "Threat model; assess privacy; map controls; resolve gaps and conflicting decisions", "roles": "Security, privacy, compliance, risk, architecture", "deliverables": "Threat register, control matrix, risk register", "security_controls": "Least privilege, encryption, secure design review", "entry_criteria": "Requirement draft and evidence available", "exit_criteria": "High risks have owners and mitigations", "validation_gates": "Security and risk acceptance", "compliance_checkpoints": "Legal/compliance approval for interpretations"},
        {"phase": f"{top_approach} delivery planning", "activities": "Plan increments, verification mapping, dependencies, change control, and release gates", "roles": "Project manager, architect, product owner, QA lead", "deliverables": "Project plan, delivery workflow, verification plan", "security_controls": "Security backlog and threat model updates", "entry_criteria": "Scope and priority baselined", "exit_criteria": "Owners, estimates, milestones, and gates approved", "validation_gates": "SDLC sign-off", "compliance_checkpoints": "Compliance gates included in release plan"},
        {"phase": "Implementation and verification", "activities": "Build; peer review; SAST/dependency checks; functional, integration, resilience, and transaction-integrity tests", "roles": "Developers, QA, security team, service owners", "deliverables": "Code, test evidence, updated traceability", "security_controls": "Secure code review, secrets checks, SAST/DAST", "entry_criteria": "Approved stories and design", "exit_criteria": "Acceptance criteria pass; critical findings resolved", "validation_gates": "Test and release readiness review", "compliance_checkpoints": "Evidence package and control coverage review"},
        {"phase": "Deployment, monitoring, and improvement", "activities": "Stage deployment; monitor security, availability, incidents, and regulatory change; reassess after updates", "roles": "Operations, audit, compliance, product owner", "deliverables": "Deployment record, monitoring metrics, change and audit log", "security_controls": "Least privilege, incident response, rollback", "entry_criteria": "Release candidate and approvals complete", "exit_criteria": "Operational acceptance and review date assigned", "validation_gates": "Production approval", "compliance_checkpoints": "Compliance and audit approval before production"},
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


def _srs_document(analysis: RequirementsAnalysis, requirements: list[dict]) -> str:
    sections = ["# Software Requirements Specification", "## 1. Purpose and scope", analysis.summary, "## 2. Assumptions"]
    sections.extend(f"- {item}" for item in analysis.assumptions or ["No assumptions recorded."])
    sections.append("## 3. Requirements")
    for req in requirements:
        refs = ", ".join(req.get("source_references", [])) or req.get("reasoning", "Source not stated")
        sections.append(f"### {req['id']} - {req['category']}\n\n{req['description']}\n\n- Priority: {req['priority']}\n- Source/evidence: {refs}\n- Risk: {req['risk_level']}\n- Confidence: {req['confidence_score']}%\n- Approval: {req['approval_status']}")
        sections.extend(f"- Acceptance: {criterion}" for criterion in req["acceptance_criteria"])
    sections.append("## 4. Open issues\n\nResolve validation findings and confirm policy applicability, jurisdiction, and approvals before baselining.")
    return "\n\n".join(sections)


def _artefact_requirements(requirements: list[dict], terms: tuple[str, ...]) -> list[dict]:
    return [
        {"requirement_id": req["id"], "statement": req["description"], "sources": req.get("source_references", []), "status": "Candidate - human review required"}
        for req in requirements
        if any(term in f"{req['category']} {req['description']}".lower() for term in terms)
    ]


def _process_workflows(requirements: list[dict]) -> list[dict]:
    return [{
        "workflow_id": f"WF-{index:02d}",
        "trigger_or_requirement": req["id"],
        "actor": req["source_stakeholder"],
        "main_flow": req["description"],
        "exception_flow": "Missing information, failed verification, or policy exception is routed to an authorised human reviewer.",
        "acceptance_criteria": req["acceptance_criteria"],
        "source": req.get("source_references", []),
    } for index, req in enumerate(requirements, 1) if req["id"].startswith("FR")]


def _threat_register(requirements: list[dict]) -> list[dict]:
    rows = [
        {"id": "THREAT-01", "threat": "Prompt injection in uploaded documents", "impact": "Unsupported or unsafe requirement changes", "mitigation": "Keep sources in quoted evidence blocks; never execute document instructions; require review of low-confidence outputs."},
        {"id": "THREAT-02", "threat": "Sensitive identifiers in prompts or stored records", "impact": "Disclosure of financial or personal data", "mitigation": "Redact known patterns before retrieval/model calls; minimise and protect persisted records."},
        {"id": "THREAT-03", "threat": "Incorrect regulatory mapping", "impact": "Compliance gaps or false assurance", "mitigation": "Mark heuristic mappings as candidates and require authorised legal/compliance sign-off."},
    ]
    if any("Integration" in req["multi_labels"] for req in requirements):
        rows.append({"id": "THREAT-04", "threat": "Legacy/API compromise or data-integrity failure", "impact": "Incorrect or unavailable financial decisions", "mitigation": "Apply least privilege, authenticated interfaces, integrity checks, monitoring, and recovery tests."})
    return rows


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
        {"item": "Conflicting stakeholder decisions", "required_approver": "Business owner and compliance/risk owner", "status": "Pending"},
        {"item": "Changes to approved requirements", "required_approver": "Baseline owner plus relevant control owners", "status": "Pending"},
    ]
    for req in requirements:
        if req["risk_level"] == "High":
            gates.append({"item": req["id"], "required_approver": "Security/compliance owner", "status": req["approval_status"]})
    return gates


def _evaluation_metrics(requirements: list[dict], issues: list[dict], compliance_rows: list[dict], trace_rows: list[dict]) -> list[dict]:
    total = max(1, len(requirements))
    mapped = sum(1 for req in requirements if req["applicable_regulations"])
    traceable = sum(1 for row in trace_rows if row["evidence_status"] != "No retrieved source match")
    return [
        {"metric": "Requirement quality proxy (not benchmark completeness)", "value": f"{100 - round((len(issues) / total) * 100)}%"},
        {"metric": "Candidate evidence-link coverage (requires review)", "value": f"{round((traceable / total) * 100)}%"},
        {"metric": "Candidate control mapping coverage (requires review)", "value": f"{round((mapped / total) * 100)}%"},
        {"metric": "Compliance matrix rows", "value": str(len(compliance_rows))},
        {"metric": "Human correction queue", "value": str(len(issues))},
        {"metric": "Hallucination rate / citation correctness", "value": "Not measured - requires labelled benchmark and expert adjudication"},
        {"metric": "Extraction precision, recall, and F1", "value": "Not measured - requires ground-truth requirements"},
        {"metric": "Conflict detection accuracy", "value": "Not measured - requires labelled conflict set"},
        {"metric": "Stakeholder satisfaction and time saved", "value": "Not measured - requires user study and manual baseline"},
    ]


def _normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()
