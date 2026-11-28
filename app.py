"""
Finance Requirements Engineering Agents - Streamlit UI

Requirements, clarification, validation, and SDLC workflow, running locally
with Ollama + CrewAI. See README.md for setup.
"""

import json
import os

import streamlit as st

from src.crew_runner import (
    delete_run_history_item,
    list_run_history,
    load_audit_record,
    load_run_result,
    run_crew,
    save_approval_changes,
    save_evaluation_feedback,
    sync_local_history_to_mongo,
)
from src.db import check_mongo_connection, get_mongo_config
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

    st.divider()
    st.header("Database (MongoDB)")
    mongo_cfg = get_mongo_config()
    mongo_connected, mongo_info = check_mongo_connection()
    if mongo_connected:
        st.success(f"MongoDB connected (`{mongo_info}`).")
        if st.button("Sync local outputs/ to MongoDB", use_container_width=True):
            sync_result = sync_local_history_to_mongo()
            st.session_state["mongo_sync_notice"] = (
                f"Synced {sync_result['synced']} run(s) to MongoDB "
                f"({sync_result['skipped']} already present)."
            )
            if sync_result["errors"]:
                st.warning("Some files could not be synced: " + "; ".join(sync_result["errors"]))
        if st.session_state.get("mongo_sync_notice"):
            st.caption(st.session_state["mongo_sync_notice"])
    else:
        st.warning(
            f"MongoDB unreachable at `{mongo_cfg['uri']}`. "
            "Falling back to local `outputs/` JSON files."
        )

    history_summaries, history_backend, auto_sync_stats = list_run_history(limit=100)
    if auto_sync_stats.get("synced", 0) > 0:
        st.caption(f"Imported {auto_sync_stats['synced']} existing local run(s) into MongoDB.")

    if history_summaries:
        st.subheader(f"Recent runs ({len(history_summaries)})")
        summary_by_id = {item["run_id"]: item for item in history_summaries}
        quick_run_id = st.selectbox(
            "Load a past run",
            options=[item["run_id"] for item in history_summaries],
            format_func=lambda rid: (
                f"{rid} | {summary_by_id[rid]['model_name']} | "
                f"{summary_by_id[rid]['scenario_preview'][:42]}"
            ),
            key="sidebar_quick_run_id",
        )
        if st.button("Open selected run", use_container_width=True, key="sidebar_open_run"):
            try:
                st.session_state["last_result"] = load_run_result(quick_run_id)
                st.session_state["pending_nav_mode"] = "New / Active Analysis"
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load run {quick_run_id}: {exc}")
    else:
        st.caption("No past runs recorded yet.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Finance Requirements Engineering Agents")
st.caption("Requirements -> Clarification -> Validation -> SDLC, running locally with Ollama + CrewAI")
if AUTH_REQUIRED:
    st.caption(f"Signed in as {CURRENT_ACTOR} ({CURRENT_ROLE}).")
    if st.button("Sign out"):
        st.session_state.pop("authenticated_user", None)
        st.rerun()

if "pending_nav_mode" in st.session_state:
    st.session_state["nav_mode"] = st.session_state.pop("pending_nav_mode")
if "nav_mode" not in st.session_state:
    st.session_state["nav_mode"] = "New / Active Analysis"

nav_mode = st.radio(
    "View",
    ["New / Active Analysis", "Run History"],
    horizontal=True,
    key="nav_mode",
    label_visibility="collapsed",
)

if nav_mode == "Run History":
    st.header("Run History")
    backend_label = f"MongoDB (`{mongo_info}`)" if history_backend == "mongodb" else "Local `outputs/` files (MongoDB fallback)"
    st.caption(f"Storage backend: {backend_label}")

    if not history_summaries:
        st.info("No runs found yet. Run the agents from the New / Active Analysis view to create your first run.")
        st.stop()

    total_reqs = sum(item["functional_count"] + item["non_functional_count"] for item in history_summaries)
    unique_models = sorted({item["model_name"] for item in history_summaries if item.get("model_name")})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stored runs", len(history_summaries))
    m2.metric("Models used", len(unique_models))
    m3.metric("Total requirements", total_reqs)
    m4.metric("Latest run", history_summaries[0]["run_id"])

    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        history_search = st.text_input(
            "Search run history",
            placeholder="Filter by scenario text, run ID, SDLC approach, actor, or document name...",
            key="history_search_query",
        )
    with filter_col2:
        history_model_filter = st.selectbox(
            "Model",
            options=["All"] + unique_models,
            key="history_model_filter",
        )

    filtered_history, _, _ = list_run_history(
        limit=100,
        search_query=history_search,
        model_filter=history_model_filter,
        auto_sync_local=False,
    )

    if not filtered_history:
        st.warning("No runs match the current search or model filter.")
        st.stop()

    table_rows = [
        {
            "run_id": item["run_id"],
            "created_at": item["created_at"],
            "model": item["model_name"],
            "duration_s": round(item["duration_seconds"], 1),
            "FR": item["functional_count"],
            "NFR": item["non_functional_count"],
            "recommended_sdlc": item["recommended_approach"] or "-",
            "audit_events": item["approval_events_count"],
            "created_by": item["created_by"],
            "scenario": item["scenario_preview"],
        }
        for item in filtered_history
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.subheader("Inspect or load a past run")
    filtered_by_id = {item["run_id"]: item for item in filtered_history}
    selected_history_id = st.selectbox(
        "Select run ID",
        options=[item["run_id"] for item in filtered_history],
        format_func=lambda rid: (
            f"{rid} — {filtered_by_id[rid]['created_at']} — "
            f"{filtered_by_id[rid]['model_name']} — {filtered_by_id[rid]['scenario_preview'][:70]}"
        ),
        key="history_selected_run_id",
    )
    selected_meta = filtered_by_id[selected_history_id]

    action_col1, action_col2, action_col3 = st.columns([2, 2, 3])
    with action_col1:
        if st.button("Open full analysis", type="primary", use_container_width=True, key="history_open_full"):
            try:
                st.session_state["last_result"] = load_run_result(selected_history_id)
                st.session_state["pending_nav_mode"] = "New / Active Analysis"
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load run {selected_history_id}: {exc}")
    with action_col2:
        confirm_delete = st.checkbox("Confirm delete", key=f"confirm_del_{selected_history_id}")
    with action_col3:
        delete_allowed = (not AUTH_REQUIRED or can_run(CURRENT_ROLE)) and confirm_delete
        if st.button("Delete selected run", disabled=not delete_allowed, use_container_width=True, key="history_delete_btn"):
            deleted = delete_run_history_item(selected_history_id)
            if st.session_state.get("last_result") and st.session_state["last_result"].run_id == selected_history_id:
                st.session_state["last_result"] = None
            st.success(
                f"Deleted run {selected_history_id} "
                f"(MongoDB: {deleted['mongodb']}, file: {deleted['file']})."
            )
            st.rerun()

    with st.expander("Quick preview of selected run", expanded=True):
        st.markdown(f"**Run ID:** `{selected_meta['run_id']}` | **Created:** `{selected_meta['created_at']}` | **Actor:** `{selected_meta['created_by']}`")
        st.markdown(f"**Scenario:** {selected_meta['scenario']}")
        st.markdown(
            f"**Model:** `{selected_meta['model_name']}` | **Duration:** `{selected_meta['duration_seconds']:.1f} s` | "
            f"**FR/NFR:** `{selected_meta['functional_count']}/{selected_meta['non_functional_count']}` | "
            f"**SDLC:** `{selected_meta['recommended_approach'] or 'N/A'}`"
        )
        if selected_meta["reference_documents"]:
            st.markdown(f"**Reference documents:** {', '.join(selected_meta['reference_documents'])}")
        if selected_meta["uploaded_documents"]:
            st.markdown(f"**Uploaded documents:** {', '.join(selected_meta['uploaded_documents'])}")
        try:
            preview_record = load_audit_record(selected_history_id)
            reqs_obj = preview_record.get("requirements") or {}
            st.markdown(f"**Summary:** {reqs_obj.get('summary', '')}")
            if preview_record.get("approval_history"):
                st.caption(f"Recorded audit/approval events: {len(preview_record['approval_history'])}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read run preview: {exc}")
    st.stop()

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
    if getattr(result, "storage_info", {}).get("loaded_from_history"):
        hist_col1, hist_col2 = st.columns([5, 1])
        with hist_col1:
            st.info(
                f"Viewing historical run `{result.run_id}` "
                f"(created `{result.created_at or 'previously'}` by `{getattr(result, 'created_by', 'unknown')}`)."
            )
        with hist_col2:
            if st.button("Clear loaded run", use_container_width=True):
                st.session_state["last_result"] = None
                st.rerun()

    mongo_note = f"MongoDB (`{mongo_info}`) and " if (mongo_connected or getattr(result, "storage_info", {}).get("mongodb_saved")) else ""
    st.caption(
        f"Run {result.run_id} with `{result.model_name}` took {result.duration_seconds:.1f} s. "
        f"Saved to {mongo_note}`outputs/run_{result.run_id}.json{'.enc' if os.getenv('FINANCE_REQ_AUDIT_ENCRYPTION_KEY') else ''}`."
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

    # ---------------- Step 4: SDLC Selection Agent ----------------
    st.header("Step 4: SDLC Selection Agent")
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

    st.header("Specialist Analysis")
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
        clarification_review = extended.get("clarification_agent")
        st.subheader("Clarification Agent questions")
        if clarification_review:
            st.caption(clarification_review.get("summary", ""))
            questions = clarification_review.get("questions", [])
            if questions:
                st.dataframe(questions, use_container_width=True, hide_index=True)
            else:
                st.success("The Clarification Agent found no material unanswered questions.")
        st.caption("Answer these with the relevant stakeholder, then include the answers in the notes and rerun the analysis.")
        st.subheader("Rule-based clarification prompts")
        for question in extended["clarification_questions"]:
            st.markdown(f"- {question}")

    with validation_tab:
        st.subheader("Classified candidate requirements")
        st.dataframe(extended["validated_requirements"], use_container_width=True, hide_index=True)
        validation_review = extended.get("validation_agent")
        if validation_review:
            st.subheader("Validation Agent review")
            st.caption(validation_review.get("summary", ""))
            if validation_review.get("status") != "pass" or validation_review.get("findings"):
                st.warning("The Validation Agent found items that need human review.")
            else:
                st.success("The Validation Agent found no material issues; human approval is still required.")
            findings = validation_review.get("findings", [])
            if findings:
                st.dataframe(findings, use_container_width=True, hide_index=True)
            st.caption(
                f"Requirements reviewed: {', '.join(validation_review.get('reviewed_requirement_ids', [])) or 'None reported'}"
            )
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
            data=json.dumps({"requirements": req.model_dump(), "clarification_agent": extended.get("clarification_agent"), "validation_agent": extended.get("validation_agent"), "sdlc": sdlc.model_dump(), "artefacts": artifacts, "traceability": extended["traceability_matrix"], "compliance_matrix": extended["compliance_matrix"], "fairness_review": extended["fairness_review"], "regulatory_change_controls": extended["regulatory_change_controls"], "risks": extended["risk_register"]}, indent=2, ensure_ascii=False),
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
