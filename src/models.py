"""
Structured data models for the Finance Requirements Engineering Agents.

These Pydantic models are passed to CrewAI as `output_pydantic` schemas so
that each agent is forced to return clean, structured JSON instead of free
text. Structured outputs let the Streamlit UI render results and let later
agents reliably consume earlier workflow outputs.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, model_validator


class RequirementItem(BaseModel):
    """A single functional or non-functional requirement."""

    id: str = Field(..., description="Short identifier, e.g. FR-01 or NFR-01")
    category: str = Field(..., description="e.g. Application Intake, Security, Availability")
    description: str = Field(..., description="The requirement statement, phrased as 'The system shall ...'")
    priority: str = Field(..., description="High, Medium, or Low")
    reasoning: str = Field(..., description="Why this requirement exists, citing the source document/section if applicable")
    source_stakeholder: str = Field(default="Unspecified stakeholder", description="Stakeholder or source that motivated the requirement")
    dependencies: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    applicable_regulations: List[str] = Field(default_factory=list)
    risk_level: str = Field(default="Medium", description="High, Medium, or Low")
    confidence_score: int = Field(default=70, ge=0, le=100)
    approval_status: str = Field(default="Pending human approval")
    source_references: List[str] = Field(
        default_factory=list,
        description="Exact source names and section/page references supporting this requirement",
    )


class RequirementsAnalysis(BaseModel):
    """Full output of the Requirements Agent (Step 1)."""

    summary: str = Field(..., description="One or two sentence summary of the scenario")
    functional_requirements: List[RequirementItem] = Field(default_factory=list)
    non_functional_requirements: List[RequirementItem] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class ClarificationQuestion(BaseModel):
    """A stakeholder question generated to resolve an open requirement issue."""

    question: str
    reason: str
    related_requirement_ids: List[str] = Field(default_factory=list)
    priority: Literal["High", "Medium", "Low"] = "Medium"


class ClarificationAnalysis(BaseModel):
    """Reviewable questions raised by the Clarification Agent."""

    summary: str
    questions: List[ClarificationQuestion] = Field(default_factory=list)


class ValidationFinding(BaseModel):
    """One evidence-based issue found while reviewing candidate requirements."""

    requirement_id: str = Field(
        description="Requirement ID, or 'General' for a cross-requirement finding"
    )
    issue_type: Literal[
        "ambiguity",
        "incomplete",
        "inconsistent",
        "duplicate",
        "not_testable",
        "source_traceability",
        "other",
    ]
    severity: Literal["High", "Medium", "Low"]
    description: str
    evidence: str = Field(
        description="Requirement text or source evidence supporting the finding"
    )
    recommended_action: str


class RequirementsValidation(BaseModel):
    """AI-assisted validation results; findings require human review."""

    status: Literal["pass", "needs_review"]
    summary: str
    reviewed_requirement_ids: List[str] = Field(default_factory=list)
    findings: List[ValidationFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_review_for_findings(self):
        if self.findings:
            self.status = "needs_review"
        return self


class SDLCRecommendation(BaseModel):
    """One ranked SDLC candidate."""

    approach: str = Field(..., description="e.g. Agile, DevSecOps, Spiral, V-Model, Waterfall, or a named hybrid")
    confidence_percent: int = Field(..., ge=0, le=100)
    justification: str = Field(..., description="Why this approach fits, referencing project characteristics")


class SDLCAnalysis(BaseModel):
    """Full output of the SDLC Selection Agent."""

    project_characteristics: List[str] = Field(
        default_factory=list,
        description="Derived characteristics, e.g. 'High regulatory criticality', 'Stable requirements'",
    )
    ranked_recommendations: List[SDLCRecommendation] = Field(default_factory=list)
    recommended_approach: str = Field(..., description="The top-ranked SDLC approach")
    workflow_notes: str = Field(
        ..., description="Phases/activities/checkpoints suggested for this project (short paragraph or bullet text)"
    )
    approval_note: str = Field(
        default="This recommendation requires sign-off from the project manager, architect, "
        "security lead, and compliance officer before adoption.",
        description="Human-in-the-loop reminder",
    )
    decision_factors: List[str] = Field(
        default_factory=list,
        description="Project characteristics used to rank the candidate SDLC models",
    )
