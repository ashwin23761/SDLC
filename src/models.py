"""
Structured data models for the Finance Requirements Engineering Agents.

These Pydantic models are passed to CrewAI as `output_pydantic` schemas so
that each agent is forced to return clean, structured JSON instead of free
text. That structured output is what lets the Streamlit UI render tables
and lets Step 2 (the SDLC agent) reliably consume Step 1's output.
"""

from typing import List
from pydantic import BaseModel, Field


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


class RequirementsAnalysis(BaseModel):
    """Full output of the Requirements Agent (Step 1)."""

    summary: str = Field(..., description="One or two sentence summary of the scenario")
    functional_requirements: List[RequirementItem] = Field(default_factory=list)
    non_functional_requirements: List[RequirementItem] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class SDLCRecommendation(BaseModel):
    """One ranked SDLC candidate."""

    approach: str = Field(..., description="e.g. Agile, DevSecOps, Spiral, V-Model, Waterfall, or a named hybrid")
    confidence_percent: int = Field(..., ge=0, le=100)
    justification: str = Field(..., description="Why this approach fits, referencing project characteristics")


class SDLCAnalysis(BaseModel):
    """Full output of the SDLC Agent (Step 2)."""

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
