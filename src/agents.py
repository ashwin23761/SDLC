"""
CrewAI agent definitions for the requirements-analysis workflow.

This is a deliberately small slice of the full multi-agent architecture in
the problem statement (Coordinator, Stakeholder-interaction, Extraction,
Clarification, Classification, Conflict-detection, Compliance, Security,
Risk, SDLC-selection, Documentation, Validation, Human-approval agents).
The active LLM workflow implements requirements, clarification, validation,
and SDLC selection agents; deterministic specialists handle additional checks.
"""

from crewai import Agent, LLM


def build_llm(model_name: str, base_url: str = "http://localhost:11434") -> LLM:
    """
    Build a CrewAI LLM object pointed at a local Ollama server.
    CrewAI (via litellm) expects Ollama models prefixed with 'ollama/'.
    """
    prefixed = model_name if model_name.startswith("ollama/") else f"ollama/{model_name}"
    return LLM(model=prefixed, base_url=base_url, temperature=0.2)


def build_requirements_agent(llm: LLM) -> Agent:
    return Agent(
        role="Financial Software Requirements Analyst",
        goal=(
            "Read a plain-language description of a financial-sector functionality and the supplied "
            "reference/policy documents, then produce a complete, well-structured set of functional and "
            "non-functional requirements grounded in those documents."
        ),
        backstory=(
            "You are a senior business analyst who has spent a decade gathering software requirements "
            "for retail and commercial banks. You are precise, you never invent regulations, and whenever "
            "a requirement is driven by a policy or KYC/AML rule you cite the document and section it came "
            "from. You always separate functional requirements (what the system must do) from "
            "non-functional requirements (security, availability, compliance, auditability, performance)."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_clarification_agent(llm: LLM) -> Agent:
    return Agent(
        role="Financial Requirements Clarification Analyst",
        goal=(
            "Review the scenario, extracted requirements, and supplied evidence to identify material "
            "ambiguities, missing decisions, and unresolved stakeholder questions."
        ),
        backstory=(
            "You are a senior requirements interviewer for financial systems. Ask concise, answerable "
            "questions that resolve scope, business-rule, exception, data, integration, security, "
            "compliance, performance, or availability uncertainty. Do not ask for information already "
            "provided, invent policy, or answer questions on behalf of stakeholders."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_validation_agent(llm: LLM) -> Agent:
    return Agent(
        role="Financial Requirements Validation Analyst",
        goal=(
            "Check candidate requirements for completeness, clarity, testability, internal consistency, "
            "duplication, and traceability to the supplied scenario and source evidence."
        ),
        backstory=(
            "You are an independent requirements quality reviewer for regulated financial software. "
            "Report concrete findings with requirement IDs and supporting evidence. Treat missing "
            "evidence as unverified, not as proof that a requirement is false. Do not rewrite or approve "
            "requirements, make legal determinations, or treat AI output as a human-approved baseline."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_sdlc_agent(llm: LLM) -> Agent:
    return Agent(
        role="Software Development Life Cycle (SDLC) Selection Advisor",
        goal=(
            "Analyse a validated set of requirements for a financial system and recommend the most "
            "suitable SDLC model(s), ranked with confidence scores and justification."
        ),
        backstory=(
            "You are a solutions architect who specialises in matching software development methodologies "
            "(Waterfall, V-Model, Spiral, Agile, DevSecOps, or hybrids) to project risk, regulatory "
            "criticality, and requirement volatility. You reason from the requirements handed to you, "
            "extract implicit project characteristics (e.g. how regulated, how likely to change, how "
            "severe a failure would be), and never recommend a single option without alternatives and a "
            "clear rationale. You always note that the final choice needs human sign-off."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
