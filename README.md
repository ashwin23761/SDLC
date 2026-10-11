# Finance Requirements Engineering Agents

An Agentic-AI prototype for automated requirement gathering, analysis, and SDLC
identification in the financial sector.

The app combines two local LLM/CrewAI agents with deterministic specialist agents/checkers:

1. **Requirements Agent** - extracts functional and non-functional requirements from a
   financial scenario, selected knowledge-base files, and uploaded source documents.
2. **SDLC Agent** - derives project characteristics and recommends ranked SDLC models.
3. **Clarification and validation checkers** - detect missing, vague, duplicate, or
   weakly evidenced requirements and generate adaptive follow-up questions.
4. **Classification, compliance, security/privacy, and risk checkers** - multi-label
   requirements and map them to controls, risks, and human approval checkpoints.
5. **Documentation and traceability generators** - produce user stories, use cases,
   acceptance criteria, an assumptions/dependencies register, open issues, and a
   Requirements Traceability Matrix.
6. **Hybrid SDLC decision engine** - combines rules with the LLM recommendation to
   produce ranked scores and a project-specific workflow.
7. **Human approval workspace** - lets reviewers accept, reject, or request regeneration
   for critical outputs.

It runs locally with Ollama for the LLM, CrewAI for orchestration, and Streamlit for the UI.

## Project structure

```text
finance-req-agents/
|-- app.py                     # Streamlit UI
|-- requirements.txt
|-- src/
|   |-- agents.py              # CrewAI Agent definitions
|   |-- analysis_engine.py     # Rule-based specialist agents/checkers
|   |-- tasks.py               # CrewAI Task definitions
|   |-- crew_runner.py         # Orchestrates the two-agent crew
|   |-- models.py              # Pydantic schemas
|   |-- rag.py                 # Upload parsing, chunking, and retrieval
|   |-- document_loader.py     # Loads built-in documents
|   `-- ollama_client.py       # Checks local Ollama status/model availability
|-- documents/                 # Reference/policy documents
`-- outputs/                   # Auto-saved JSON audit records
```

## Setup

1. Install Ollama and pull a small local model:

   ```bash
   ollama pull llama3.2:3b
   ollama serve
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   cd finance-req-agents
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

   It opens at `http://localhost:8501`.

## Using it

1. In the sidebar, confirm Ollama is running and your selected model is available.
2. Tick which built-in reference documents should be searchable by the Requirements Agent.
3. Optionally upload project/evaluation sources (`.pdf`, `.md`, or `.txt`) such as the
   problem statement or course-evaluation email text.
4. Describe the financial functionality in the text box.
5. Click **Run agents**. The app retrieves the most relevant chunks from selected and
   uploaded sources, runs the Requirements Agent, passes the structured output to the
   SDLC Agent, then runs the deterministic specialist analysis agents.
6. Review the Step 3 tabs for input governance, validation, compliance/risk, generated
   artefacts, traceability, SDLC scoring, and human approval controls.
7. Each run is saved as `outputs/run_<timestamp>.json` for traceability/audit purposes.

## RAG upload flow

The app includes a lightweight local RAG layer in `src/rag.py`:

1. Selected knowledge-base files and uploaded `.pdf`, `.md`, or `.txt` files are parsed.
2. Documents are split into overlapping chunks with source labels.
3. Chunks are ranked against the user's prompt using local keyword/TF-IDF-style scoring.
4. The top chunks are injected into the Requirements Agent task as evidence.

Uploaded files are treated as source material only, not as instructions to the app. This is
important for evaluation artifacts: the problem statement and email define what the project
should demonstrate, while the user's prompt remains the runtime request.

## Adding your own reference documents

Drop any `.md` or `.txt` file into `documents/`. Write real prose with section numbers
(for example, `## 2.1 Credit bureau report`) so the Requirements Agent can cite the
document name and section in the reasoning column.

## Mapping to the problem statement

| Evaluation requirement | Where it is implemented |
|---|---|
| Multi-agent architecture and central orchestration | `src/crew_runner.py` runs CrewAI agents, then specialist agents in `src/analysis_engine.py` |
| Stakeholder/source inputs | Streamlit scenario box, selected knowledge-base files, and upload controls in `app.py` |
| Retrieval-grounded generation | `src/rag.py` chunks and ranks selected/uploaded sources and frames them as evidence only |
| Sensitive-data handling | `analysis_engine.mask_sensitive_data` masks common email, phone, card-like, Aadhaar-like, and tax-ID patterns |
| Requirement extraction | `src/tasks.py::build_requirements_task` -> `RequirementsAnalysis` |
| Requirement structure | `src/models.py::RequirementItem` includes ID, statement, category, source, priority, dependencies, acceptance criteria, regulations, risk, confidence, and approval status |
| Classification | `analysis_engine._enrich_requirement` assigns business, functional, security, privacy, regulatory, performance, availability, auditability, integration, and operations labels |
| Ambiguity/incompleteness/duplication/testability checks | `analysis_engine._quality_issues` and the Validation tab |
| Compliance and security analysis | `analysis_engine._compliance_matrix`, `_security_privacy_controls`, and `_risk_register` |
| Generated artefacts | Step 3 Artefacts tab: user stories, use cases, acceptance criteria, assumptions/dependencies, open issues |
| Traceability | Step 3 Traceability tab and JSON audit record |
| SDLC decision factors and ranked recommendations | CrewAI SDLC task plus `analysis_engine._score_sdlc` |
| Project-specific SDLC workflow | Step 3 SDLC engine tab |
| Human-in-the-loop approval | Step 3 Human approval tab with editable accept/reject/regenerate statuses |
| Audit trail | `outputs/run_<timestamp>.json` stores prompts, sources, retrieved chunks, requirements, SDLC result, specialist analysis, and approval queue seed |

## Notes / known limitations

- Structured output quality depends on the local model. `llama3.2:3b` is fast but can be
  inconsistent with strict JSON; if parsing fails, try `llama3.1:8b` or `qwen2.5:7b`.
- Compliance mappings are prototype controls, not legal determinations. The UI explicitly
  routes high-impact interpretations to authorised humans.
- Retrieval is intentionally lightweight and local. It can be upgraded to embeddings or a
  vector database while keeping the same source-document interface.
