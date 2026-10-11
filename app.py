"""
Finance Requirements Engineering Agents - Streamlit UI

Phase 1 prototype: Requirements Agent -> SDLC Agent, running locally with
Ollama + CrewAI. See README.md for setup.
"""

import json

import streamlit as st

from src.crew_runner import run_crew
from src.document_loader import list_documents
from src.ollama_client import check_ollama
from src.rag import SourceDocument, extract_text_from_upload

st.set_page_config(page_title="Finance Requirements Agents", layout="wide")

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_SCENARIO = (
    "Loan approval module for a retail bank: customers apply online for a personal loan, "
    "the system checks eligibility and credit score, and either auto-decides or routes the "
    "application to a human underwriter."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    model_name = st.text_input(
        "Ollama model",
        value=st.session_state.get("model_name", "llama3.2:3b"),
        help="Name of a model already pulled with `ollama pull <model>`.",
    )
    st.session_state["model_name"] = model_name

    base_url = st.text_input("Ollama base URL", value=DEFAULT_BASE_URL)

    server_running, model_available, all_models = check_ollama(base_url, model_name)
    if server_running and model_available:
        st.success(f"Ollama is running and **{model_name}** is available.")
    elif server_running and not model_available:
        st.warning(
            f"Ollama is running, but **{model_name}** was not found.\n\n"
            f"Available models: {', '.join(all_models) if all_models else '(none pulled yet)'}\n\n"
            f"Run `ollama pull {model_name}` to fetch it."
        )
    else:
        st.error(
            "Could not reach Ollama at this URL. Make sure it's running "
            "(`ollama serve`) and reachable at the URL above."
        )

    st.divider()
    st.header("Reference documents")
    st.caption("Give these to the Requirements Agent")

    available_docs = list_documents()
    selected_docs = []
    if not available_docs:
        st.info("No .md/.txt files found in documents/.")
    else:
        for doc in available_docs:
            if st.checkbox(doc, value=True, key=f"doc_{doc}"):
                selected_docs.append(doc)

    st.caption("Files are read from the `documents/` folder.")

    st.divider()
    st.header("Upload RAG sources")
    uploaded_files = st.file_uploader(
        "Upload problem statements, emails, or reference notes",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        help="Uploaded files are retrieved against your prompt and passed to the Requirements Agent as source context.",
    )
    retrieval_top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=10,
        value=6,
        help="Higher values give the agent more source context, but very small local models may become less consistent.",
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Finance Requirements Engineering Agents")
st.caption("Phase 1: Requirements Agent -> SDLC Agent, running locally with Ollama + CrewAI")

scenario = st.text_area(
    "Describe the finance functionality",
    value=st.session_state.get("scenario", DEFAULT_SCENARIO),
    height=110,
)
st.session_state["scenario"] = scenario

run_clicked = st.button("Run agents", type="primary", disabled=not (server_running and model_available))
if not (server_running and model_available):
    st.caption("Fix the Ollama status in the sidebar before running.")

if run_clicked:
    with st.spinner(f"Running Requirements Agent, then SDLC Agent, using {model_name}..."):
        try:
            uploaded_documents = []
            for uploaded in uploaded_files or []:
                text = extract_text_from_upload(uploaded.name, uploaded.getvalue())
                uploaded_documents.append(
                    SourceDocument(name=uploaded.name, text=text, source_type="upload")
                )

            result = run_crew(
                scenario=scenario,
                reference_documents=selected_docs,
                model_name=model_name,
                base_url=base_url,
                uploaded_documents=uploaded_documents,
                retrieval_top_k=retrieval_top_k,
            )
            st.session_state["last_result"] = result
        except Exception as e:  # noqa: BLE001 - surface any crew/LLM error to the user
            st.error(f"Run failed: {e}")
            st.session_state["last_result"] = None

result = st.session_state.get("last_result")

if result:
    st.divider()
    st.caption(
        f"Run {result.run_id} with `{result.model_name}` took {result.duration_seconds:.1f} s. "
        f"Saved to `outputs/run_{result.run_id}.json`."
    )
    if result.uploaded_documents:
        st.caption(f"Uploaded sources: {', '.join(result.uploaded_documents)}")
    if result.retrieved_chunks:
        with st.expander("Retrieved RAG context"):
            for chunk in result.retrieved_chunks:
                st.markdown(
                    f"**{chunk.source_name}** - chunk {chunk.chunk_index} - "
                    f"score {chunk.score:.3f}"
                )
                st.caption(chunk.text[:600] + ("..." if len(chunk.text) > 600 else ""))

    # ---------------- Step 1: Requirements Agent ----------------
    st.header("Step 1: Requirements Agent")
    req = result.requirements
    extended = result.extended_analysis

    st.markdown(f"**Summary:** {req.summary}")

    st.subheader(f"Functional requirements ({len(req.functional_requirements)})")
    if req.functional_requirements:
        st.dataframe(
            [r.model_dump() for r in req.functional_requirements],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("None generated.")

    st.subheader(f"Non-functional requirements ({len(req.non_functional_requirements)})")
    if req.non_functional_requirements:
        st.dataframe(
            [r.model_dump() for r in req.non_functional_requirements],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("None generated.")

    st.subheader("Assumptions")
    if req.assumptions:
        for a in req.assumptions:
            st.markdown(f"- {a}")
    else:
        st.caption("None stated.")

    # ---------------- Step 2: SDLC Agent ----------------
    st.header("Step 2: SDLC Agent")
    sdlc = result.sdlc

    st.subheader("Project characteristics")
    for c in sdlc.project_characteristics:
        st.markdown(f"- {c}")

    st.subheader("Ranked SDLC recommendations")
    if sdlc.ranked_recommendations:
        st.dataframe(
            [r.model_dump() for r in sdlc.ranked_recommendations],
            use_container_width=True,
            hide_index=True,
        )
    st.success(f"**Recommended approach:** {sdlc.recommended_approach}")

    st.subheader("Suggested workflow")
    st.markdown(sdlc.workflow_notes)

    st.info(sdlc.approval_note)

    st.header("Step 3: Specialist Analysis Agents")
    (
        governance_tab,
        validation_tab,
        compliance_tab,
        artifacts_tab,
        traceability_tab,
        sdlc_tab,
        approval_tab,
    ) = st.tabs(
        [
            "Input governance",
            "Validation",
            "Compliance & risk",
            "Artefacts",
            "Traceability",
            "SDLC engine",
            "Human approval",
        ]
    )

    with governance_tab:
        st.subheader("Sensitive-data handling and source control")
        governance = extended["input_governance"]
        if governance["masking_events"]:
            st.dataframe(governance["masking_events"], use_container_width=True, hide_index=True)
        else:
            st.caption("No obvious sensitive identifiers were found in the scenario text.")
        st.markdown(f"**Source allowlist:** {governance['source_allowlist_note']}")
        st.markdown("**Masked scenario preview**")
        st.code(governance["masked_scenario"], language="text")

        st.subheader("Role-specific interview templates")
        st.dataframe(extended["stakeholder_interview_templates"], use_container_width=True, hide_index=True)
        st.subheader("Adaptive clarification questions")
        for question in extended["clarification_questions"]:
            st.markdown(f"- {question}")

    with validation_tab:
        st.subheader("Validated and classified requirements")
        st.dataframe(extended["validated_requirements"], use_container_width=True, hide_index=True)
        st.subheader("Requirement quality issues")
        if extended["quality_issues"]:
            st.dataframe(extended["quality_issues"], use_container_width=True, hide_index=True)
        else:
            st.success("No rule-based ambiguity, duplication, or testability issues were detected.")

    with compliance_tab:
        st.subheader("Compliance-control matrix")
        st.dataframe(extended["compliance_matrix"], use_container_width=True, hide_index=True)
        st.subheader("Security and privacy controls")
        st.dataframe(extended["security_privacy_controls"], use_container_width=True, hide_index=True)
        st.subheader("Risk register")
        st.dataframe(extended["risk_register"], use_container_width=True, hide_index=True)

    with artifacts_tab:
        artifacts = extended["generated_artifacts"]
        st.subheader("User stories")
        st.dataframe(artifacts["user_stories"], use_container_width=True, hide_index=True)
        st.subheader("Use cases")
        st.dataframe(artifacts["use_cases"], use_container_width=True, hide_index=True)
        st.subheader("Acceptance criteria")
        st.dataframe(artifacts["acceptance_criteria"], use_container_width=True, hide_index=True)
        st.subheader("Assumptions and dependencies")
        st.dataframe(artifacts["assumptions_and_dependencies"], use_container_width=True, hide_index=True)
        st.subheader("Open issues")
        for issue in artifacts["open_issues"]:
            st.markdown(f"- {issue}")

    with traceability_tab:
        st.subheader("Requirements Traceability Matrix")
        st.dataframe(extended["traceability_matrix"], use_container_width=True, hide_index=True)
        st.subheader("Evaluation metrics")
        st.dataframe(extended["evaluation_metrics"], use_container_width=True, hide_index=True)

    with sdlc_tab:
        engine = extended["sdlc_decision_engine"]
        st.subheader("Decision factors")
        st.dataframe(engine["decision_factors"], use_container_width=True, hide_index=True)
        st.subheader("Rule-based SDLC scores")
        st.dataframe(engine["rule_scores"], use_container_width=True, hide_index=True)
        st.subheader("Project-specific workflow")
        st.dataframe(engine["workflow"], use_container_width=True, hide_index=True)

    with approval_tab:
        st.subheader("Human-in-the-loop approval queue")
        edited_gates = st.data_editor(
            extended["human_approval_gates"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "status",
                    options=["Pending", "Approved", "Rejected", "Needs regeneration", "Pending human approval"],
                )
            },
            key=f"approval_editor_{result.run_id}",
        )
        st.caption(
            "This workspace demonstrates accept/reject/modify/regenerate control. "
            "The original AI output and this edited approval queue are kept separate for auditability."
        )
        if st.download_button(
            "Download approval queue JSON",
            data=json.dumps(edited_gates, indent=2),
            file_name=f"approval_queue_{result.run_id}.json",
            mime="application/json",
        ):
            st.toast("Approval queue downloaded.")
