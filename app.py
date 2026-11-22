"""
Finance Requirements Engineering Agents - Streamlit UI

Phase 1 prototype: Requirements Agent -> SDLC Agent, running locally with
Ollama + CrewAI. See README.md for setup.
"""

import json
import os

import streamlit as st

from src.crew_runner import run_crew, save_approval_changes, load_audit_record, save_evaluation_feedback
from src.evaluation import evaluate_against_reference
from src.document_loader import list_documents
from src.ollama_client import check_ollama
from src.rag import SourceDocument, extract_text_from_upload
from src.security import AUTH_REQUIRED, load_users, verify_login, can_run, can_approve

st.set_page_config(page_title="Finance Requirements Agents", layout="wide")

if AUTH_REQUIRED:
    encryption_key = os.getenv("FINANCE_REQ_AUDIT_ENCRYPTION_KEY", "")
    if not encryption_key:
        st.error("Secure mode requires FINANCE_REQ_AUDIT_ENCRYPTION_KEY so run records are encrypted at rest.")
        st.stop()
    try:
        from cryptography.fernet import Fernet
        Fernet(encryption_key.encode("ascii"))
    except Exception as exc:
        st.error(f"The configured audit encryption key is invalid or cryptography is unavailable: {exc}")
        st.stop()
    users = load_users()
    if not users:
        st.error("Secure mode is enabled but no valid user registry was found. Configure FINANCE_REQ_USERS_FILE and add a user with python -m src.security add-user.")
        st.stop()
    if not st.session_state.get("authenticated_user"):
        st.title("Finance Requirements Agents - Sign in")
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            totp_code = st.text_input("Authenticator code", max_chars=6)
            submitted = st.form_submit_button("Sign in")
        if submitted:
            identity = verify_login(users, username, password, totp_code)
            if identity:
                st.session_state["authenticated_user"] = identity
                st.rerun()
            st.error("Invalid credentials or authenticator code.")
        st.stop()
    CURRENT_USER = st.session_state["authenticated_user"]
    CURRENT_ROLE = CURRENT_USER["role"]
    CURRENT_ACTOR = CURRENT_USER["username"]
else:
    CURRENT_USER = {"username": "local-unsecured-user", "role": "unsecured_local"}
    CURRENT_ROLE = CURRENT_USER["role"]
    CURRENT_ACTOR = CURRENT_USER["username"]

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
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
        value=st.session_state.get("model_name", DEFAULT_MODEL_NAME),
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

    st.caption("Supported knowledge-base files are read from the `documents/` folder.")

    st.divider()
    st.header("Upload RAG sources")
    uploaded_files = st.file_uploader(
        "Upload problem statements, emails, or reference notes",
        type=["pdf", "docx", "md", "txt", "csv", "json", "yaml", "yml", "xml", "log", "html", "htm"],
        accept_multiple_files=True,
        help="PDF, DOCX, Markdown, text, CSV, JSON, YAML, XML, log, and HTML sources are retrieved against your scenario.",
    )
    upload_metadata = {}
    for uploaded in uploaded_files or []:
        with st.expander(f"Source metadata: {uploaded.name}"):
            upload_metadata[uploaded.name] = {
                "jurisdiction": st.text_input("Jurisdiction (or Unspecified)", value="Unspecified", key=f"jurisdiction_{uploaded.name}"),
                "effective_date": st.text_input("Effective date (or Unknown)", value="Unknown", key=f"effective_{uploaded.name}"),
                "version": st.text_input("Version (or Unknown)", value="Unknown", key=f"version_{uploaded.name}"),
                "owner": st.text_input("Document owner (or Unknown)", value="Unknown", key=f"owner_{uploaded.name}"),
                "applicability": st.text_input("Applicability", value="User-provided source; applicability unverified", key=f"applicability_{uploaded.name}"),
                "review_status": "User-provided; unverified",
            }
    stakeholder_role = st.selectbox(
        "Stakeholder role for interview notes",
        ["Customer/end user", "Business analyst", "Product owner", "Architect/developer", "Security", "Compliance/legal", "Risk", "Operations/audit", "Auditor/regulator", "Other"],
    )
    stakeholder_notes = st.text_area("Interview transcripts, meeting notes, or questionnaire responses", height=100)
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
if AUTH_REQUIRED:
    st.caption(f"Signed in as {CURRENT_ACTOR} ({CURRENT_ROLE}).")
    if st.button("Sign out"):
        st.session_state.pop("authenticated_user", None)
        st.rerun()
else:
    st.warning("Local prototype mode: authentication/MFA is disabled. Enable secure mode before shared or remote use.")

scenario = st.text_area(
    "Describe the finance functionality",
    value=st.session_state.get("scenario", DEFAULT_SCENARIO),
    height=110,
)
st.session_state["scenario"] = scenario
analysis_prompt = scenario
if stakeholder_notes.strip():
    analysis_prompt += f"\n\nStakeholder statement ({stakeholder_role}):\n{stakeholder_notes.strip()}"

run_permission = not AUTH_REQUIRED or can_run(CURRENT_ROLE)
approval_permission = not AUTH_REQUIRED or can_approve(CURRENT_ROLE)
run_clicked = st.button("Run agents", type="primary", disabled=not (server_running and model_available and run_permission))
if AUTH_REQUIRED and not run_permission:
    st.info("Your role is read-only and cannot start an analysis.")
if not (server_running and model_available):
    st.caption("Fix the Ollama status in the sidebar before running.")

if run_clicked:
    with st.spinner(f"Running Requirements Agent, then SDLC Agent, using {model_name}..."):
        try:
            uploaded_documents = []
            for uploaded in uploaded_files or []:
                text = extract_text_from_upload(uploaded.name, uploaded.getvalue())
                uploaded_documents.append(
                    SourceDocument(name=uploaded.name, text=text, source_type="upload", metadata=upload_metadata.get(uploaded.name, {}))
                )

            result = run_crew(
                scenario=analysis_prompt,
                reference_documents=selected_docs,
                model_name=model_name,
                base_url=base_url,
                uploaded_documents=uploaded_documents,
                retrieval_top_k=retrieval_top_k,
                actor=CURRENT_ACTOR,
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
        f"Saved to `outputs/run_{result.run_id}.json{'.enc' if os.getenv('FINANCE_REQ_AUDIT_ENCRYPTION_KEY') else ''}`."
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
    extended = dict(result.extended_analysis)
    saved_record = {}
    try:
        saved_record = load_audit_record(result.run_id)
        extended["human_approval_gates"] = saved_record.get("extended_analysis", {}).get(
            "human_approval_gates", extended["human_approval_gates"]
        )
    except (OSError, ValueError):
        pass
    human_modified_requirements = saved_record.get("human_modified_requirements", {})

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
        if governance.get("source_metadata"):
            st.subheader("Retrieved source metadata and applicability")
            metadata_rows = [{"source": name, **metadata} for name, metadata in governance["source_metadata"].items()]
            st.dataframe(metadata_rows, use_container_width=True, hide_index=True)
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
        st.subheader("Fairness and bias review")
        st.dataframe(extended["fairness_review"], use_container_width=True, hide_index=True)
        st.subheader("Regulatory change and source currency")
        st.dataframe(extended["regulatory_change_controls"], use_container_width=True, hide_index=True)

    with artifacts_tab:
        artifacts = extended["generated_artifacts"]
        st.subheader("Draft Software Requirements Specification")
        st.download_button("Download SRS (Markdown)", artifacts["srs"], file_name=f"srs_{result.run_id}.md", mime="text/markdown")
        st.download_button(
            "Download all generated artefacts (JSON)",
            data=json.dumps({"requirements": req.model_dump(), "sdlc": sdlc.model_dump(), "artefacts": artifacts, "traceability": extended["traceability_matrix"], "compliance_matrix": extended["compliance_matrix"], "fairness_review": extended["fairness_review"], "regulatory_change_controls": extended["regulatory_change_controls"], "risks": extended["risk_register"]}, indent=2, ensure_ascii=False),
            file_name=f"requirements_package_{result.run_id}.json",
            mime="application/json",
        )
        st.text_area("SRS preview", artifacts["srs"], height=320, disabled=True)
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
        st.subheader("Process workflows")
        st.dataframe(artifacts["process_workflows"], use_container_width=True, hide_index=True)
        st.subheader("Data requirements")
        st.dataframe(artifacts["data_requirements"], use_container_width=True, hide_index=True)
        st.subheader("Interface and integration requirements")
        st.dataframe(artifacts["interface_requirements"], use_container_width=True, hide_index=True)
        st.subheader("Threat register")
        st.dataframe(artifacts["threat_register"], use_container_width=True, hide_index=True)

    with traceability_tab:
        st.subheader("Requirements Traceability Matrix")
        st.dataframe(extended["traceability_matrix"], use_container_width=True, hide_index=True)
        st.subheader("Evaluation metrics")
        st.dataframe(extended["evaluation_metrics"], use_container_width=True, hide_index=True)
        st.caption("Measured accuracy metrics require a human-labelled reference set; proxy figures are clearly marked.")
        benchmark = st.file_uploader("Optional labelled benchmark JSON", type=["json"], key=f"benchmark_{result.run_id}")
        if benchmark is not None:
            try:
                benchmark_data = json.loads(benchmark.getvalue().decode("utf-8"))
                measured = evaluate_against_reference(load_audit_record(result.run_id), benchmark_data)
                st.subheader("Benchmark comparison")
                st.warning(measured["warning"])
                st.json({key: value for key, value in measured.items() if key != "alignment"})
                st.dataframe(measured["alignment"], use_container_width=True, hide_index=True)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OSError) as exc:
                st.error(f"Benchmark could not be evaluated: {exc}")
        st.subheader("Reviewer feedback")
        with st.form(f"feedback_form_{result.run_id}"):
            satisfaction = st.slider("Stakeholder satisfaction", min_value=1, max_value=5, value=3)
            manual_minutes = st.number_input("Estimated manual analysis time (minutes)", min_value=0, value=0)
            feedback_notes = st.text_area("Corrections or feedback")
            feedback_submit = st.form_submit_button("Save evaluation feedback")
        if feedback_submit:
            feedback = {
                "satisfaction_1_to_5": satisfaction,
                "estimated_manual_minutes": manual_minutes or None,
                "system_processing_seconds": result.duration_seconds,
                "reviewer_notes": feedback_notes,
                "actor": CURRENT_ACTOR,
            }
            try:
                digest = save_evaluation_feedback(result.run_id, feedback, CURRENT_ACTOR)
                st.success(f"Feedback saved to the run audit history. Event hash: {digest[:16]}…")
            except OSError as exc:
                st.error(f"Could not save evaluation feedback: {exc}")

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
        if human_modified_requirements:
            st.subheader("Saved human modifications (separate from original AI output)")
            st.dataframe([{"item": item, **values} for item, values in human_modified_requirements.items()], use_container_width=True, hide_index=True)
        edited_gates = st.data_editor(
            [{**gate, "reviewer_notes": gate.get("reviewer_notes", ""), "modified_output": gate.get("modified_output", "")} for gate in extended["human_approval_gates"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "status",
                    options=["Pending", "Approved", "Rejected", "Needs regeneration", "Modified", "Pending human approval"],
                ),
                "reviewer_notes": st.column_config.TextColumn("reviewer_notes", help="Reason for decision or requested change"),
                "modified_output": st.column_config.TextColumn("modified_output", help="Optional human-authored replacement text"),
            },
            disabled=not approval_permission,
            key=f"approval_editor_{result.run_id}",
        )
        st.caption("Approval decisions, reviewer notes, and modifications are written to the run record with a hash-linked approval history.")
        if not approval_permission:
            st.info("Your role cannot modify or approve this queue.")
        if st.button("Save approval decisions", disabled=not approval_permission, key=f"save_approvals_{result.run_id}"):
            try:
                digest = save_approval_changes(result.run_id, edited_gates, CURRENT_ACTOR)
                st.success(f"Approval decisions saved. Event hash: {digest[:16]}…")
            except (OSError, ValueError) as exc:
                st.error(f"Could not save approvals: {exc}")
        if st.download_button(
            "Download approval queue JSON",
            data=json.dumps(edited_gates, indent=2),
            file_name=f"approval_queue_{result.run_id}.json",
            mime="application/json",
        ):
            st.toast("Approval queue downloaded.")
