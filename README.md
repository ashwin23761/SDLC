# Finance Requirements Engineering Agents

A local Streamlit prototype for requirements discovery and analysis in financial software projects. The product covers a complete advisory workflow for a first use case: source intake, retrieval, requirements extraction, deterministic specialist analysis, artefact generation, SDLC recommendations, human review, and recorded evaluation feedback.

Uploaded documents and stakeholder notes are treated as evidence, never as runtime instructions. Requirements and compliance mappings remain candidate outputs until an authorised person reviews them.

## Run locally

1. Install Ollama and pull a local model, for example `ollama pull llama3.2:3b`, then start `ollama serve`.
2. Start a local MongoDB instance (default `mongodb://localhost:27017`) or set `MONGODB_URI`, `MONGODB_DB_NAME`, and `MONGODB_COLLECTION`. If MongoDB is not running, the app falls back to reading and writing local JSON files in `outputs/`.
3. Create and activate a virtual environment, then install dependencies with `pip install -r requirements.txt`.
4. Run `streamlit run app.py` and open `http://localhost:8501`. Existing `outputs/run_*.json` files are imported into MongoDB automatically on startup (or via the sidebar sync button).

The UI supports a scenario, stakeholder interview/meeting notes, selected knowledge-base documents, and uploads in PDF, DOCX, Markdown, text, CSV, JSON, YAML, XML, log, or HTML formats. Text-based PDF extraction requires selectable text; OCR is not included.

## Run with Docker Compose

Ollama stays installed and running on the Windows host; Compose starts only the Streamlit application. Pull the configured model on the host once (the default is `llama3.2:3b`):

```powershell
ollama pull llama3.2:3b
```

From the project folder, build and start the app:

```powershell
docker compose up --build
```

Open `http://localhost:8501`. To stop the container, press Ctrl+C and run `docker compose down` if it remains running. The Compose file starts both the Streamlit app and a `mongo` (`mongo:7`) database container, persists MongoDB data in the `mongo_data` volume, keeps per-run audit JSON backups under `outputs/`, mounts `documents/` read-only, and connects to the host Ollama service at `http://host.docker.internal:11434`.

If the app reports that Ollama is unreachable, configure the host Ollama service to listen on an address reachable from Docker Desktop (commonly `OLLAMA_HOST=0.0.0.0:11434`), restart Ollama, and allow the connection through Windows Firewall. Restrict that listener with your firewall because it is reachable beyond the Windows loopback interface. To change the model or endpoint, create a `.env` file beside `docker-compose.yml`, for example:

```dotenv
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://host.docker.internal:11434
APP_PORT=8501
```

`docker compose --build` by itself is not a complete Compose command; use `docker compose up --build` to build and start in one step.

## Implemented workflow

| Problem statement capability | Implementation |
|---|---|
| Financial input and source intake | Scenario, role-labelled interview notes, selected documents, and common document/text uploads |
| Sensitive-data handling | Configured identifier patterns are masked in scenario, knowledge-base text, and uploads before retrieval or LLM submission |
| Retrieval-grounded generation | Local chunk retrieval with source labels, source metadata, prompt-injection-aware evidence framing, and explicit source selection |
| Structured requirement extraction | Functional and non-functional requirements with priorities, evidence, dependencies, acceptance criteria, controls, risk, confidence, and approval state |
| Specialist analysis | Deterministic classification, quality/conflict heuristics, candidate compliance mappings, security/privacy controls, risk and clarification queues |
| Responsible decision review | Fairness checklist for automated eligibility/credit decisions, proxy-variable review, subgroup testing, adverse-action explainability, and human appeal |
| Generated documentation | Draft Markdown SRS, user stories, use cases, acceptance criteria, process workflows, data/interface requirements, threats, assumptions, dependencies, and open issues |
| Traceability | Per-requirement source candidates, control mapping, acceptance criteria, confidence, and approval status |
| SDLC selection | LLM explanation plus deterministic ranked candidates and a workflow with phases, activities, owners, outputs, security, compliance, and validation gates |
| Human review | Editable approval queue with approve/reject/modify/regenerate statuses, reviewer notes, modifications, and saved hash-linked audit events |
| Evaluation | Transparent heuristic quality proxies, optional labelled JSON benchmark comparisons, processing time, reviewer satisfaction, and feedback capture |
| Run audit | Redacted prompt, model, retrieved evidence, generated outputs, approvals, and reviewer feedback saved to a per-run record |
| Knowledge-base metadata | `documents/manifest.json` stores jurisdiction, effective date, version, applicability, owner, and review status; missing fields are surfaced |

The workflow uses four sequential CrewAI LLM agents: requirement extraction, clarification-question generation, requirements validation, and SDLC advice. Clarification questions and validation findings are advisory outputs for stakeholder review; they do not block a run or approve requirements. Deterministic specialist analyzers in `src/analysis_engine.py` provide additional checks and artefacts. This is an orchestrated advisory prototype, not an autonomous production multi-agent service.

## Secure workspace mode

The app can require password + time-based authenticator-code login, enforce role-based run/approval permissions, and encrypt run records at rest. Create a private user registry outside the repository:

```powershell
python -m src.security add-user --file C:\secrets\finance-users.json --username admin --role admin
$env:FINANCE_REQ_USERS_FILE = 'C:\secrets\finance-users.json'
$env:FINANCE_REQ_AUDIT_ENCRYPTION_KEY = (python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())").Trim()
$env:FINANCE_REQ_AUTH_REQUIRED = '1'
streamlit run app.py
```

Supported roles include `admin`, `analyst`, `business_analyst`, `approver`, `compliance`, `security`, `architect`, `project_manager`, `operations`, `auditor`, and `viewer`. Keep the registry and encryption key in a secrets manager or protected operating-system storage, and back up the encryption key separately. Secure mode stops if the user registry or encryption key is missing or invalid.

## Benchmark input format

Upload a labelled JSON reference in the Traceability tab. Requirement matches use approximate text similarity and must be reviewed by a domain expert.

```json
{
  "requirements": [
    {
      "id": "REF-01",
      "description": "The system shall route high-value applications to a human underwriter.",
      "ambiguous": false,
      "controls": ["Loan approval policy section 3.1"]
    }
  ],
  "expected_sdlc": "Agile-V-Model hybrid"
}
```

Precision, recall, F1, ambiguity confusion counts, candidate control recall, and SDLC agreement are shown only for supplied reference labels. They do not replace adjudication or a representative study.

## Important implementation boundaries

- This repository includes illustrative sample policies only. They are not authoritative regulations. The manifest marks their jurisdiction/effective date as unspecified and unapproved; replace them with approved, versioned sources before relying on mappings.
- PII masking uses patterns for common identifiers. It cannot reliably detect every sensitive value; review inputs and use approved local models.
- Similarity-based conflict detection and source matching are screening aids and can miss or over-report issues.
- Compliance mappings are heuristic candidates and require legal/compliance sign-off. No jurisdiction-specific legal determination is made.
- Reviewer identity and role are enforced only when secure mode is configured. In default local mode the app displays a warning and permits local use without authentication.
- Audit encryption protects run records when configured. Source documents are processed in memory and are not persisted by the app. Retention/deletion policy is still an operator responsibility.
- The Streamlit app does not terminate TLS itself. Put any shared or remote deployment behind an authenticated HTTPS reverse proxy; use HTTPS for non-local model endpoints.
- User registry protection, key rotation, role lifecycle, actual regulatory updates, production monitoring, and incident response must be managed by the deployment environment. The app surfaces source currency and review actions but does not fetch changes automatically.
- Evaluation numbers are not claimed unless a labelled benchmark or reviewer feedback is supplied. The repository does not include a representative financial-sector study or manual-baseline study.

## Project structure

```text
app.py                         Streamlit workflow, run history browser, and review UI
src/db.py                      MongoDB persistence, encryption-at-rest wrapper, and history queries
src/crew_runner.py             Orchestration, redaction boundary, and audit persistence
src/analysis_engine.py         Deterministic specialist analysis and generated artefacts
src/evaluation.py              Labelled-reference comparison metrics
src/security.py                Optional local password/MFA and role checks
src/rag.py                     Document extraction, chunking, retrieval, source framing
src/models.py                  Structured requirement and SDLC schemas
src/tasks.py                   LLM task definitions
src/document_loader.py         Knowledge-base loading and manifest metadata
documents/manifest.json        Source governance metadata
outputs/                       Per-run audit records
```
