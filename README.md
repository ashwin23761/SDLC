# Finance Requirements Engineering Agents - Phase 1

An Agentic-AI application with two collaborating agents, built for the CS305 first
course-project evaluation:

1. **Requirements Agent** - identifies and lists functional and non-functional software
   requirements for a given financial scenario, grounded in retrieved reference and uploaded documents.
2. **SDLC Agent** - takes the validated requirements and recommends a suitable Software
   Development Life Cycle (SDLC) model, ranked with confidence scores and justification.

Runs locally with Ollama for the LLM, CrewAI for agent orchestration, and Streamlit for
the UI.

This is a focused slice of the larger system described in the full problem statement,
which calls for a broader multi-agent architecture backed by RAG over a financial
knowledge base. The code is structured so more agents and tasks can be added later
without restructuring the project.

## Project structure

```text
finance-req-agents/
|-- app.py                     # Streamlit UI
|-- requirements.txt
|-- src/
|   |-- agents.py              # CrewAI Agent definitions
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
   uploaded sources, then runs the Requirements Agent and passes its structured output to
   the SDLC Agent.
6. Each run is saved as `outputs/run_<timestamp>.json` for traceability/audit purposes.

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

## Mapping to the evaluation brief

| Evaluation requirement | Where it is implemented |
|---|---|
| At least two agents | `src/agents.py`: `build_requirements_agent`, `build_sdlc_agent` |
| Agent 1: identify/list software requirements for a scenario | `src/tasks.py::build_requirements_task` -> `RequirementsAnalysis` |
| Agent 2: identify a suitable SDLC for that specification | `src/tasks.py::build_sdlc_task` -> `SDLCAnalysis` |
| Agentic-AI application | CrewAI `Crew` with `Process.sequential`, run from Streamlit |
| Upload + RAG grounding | `app.py` upload control + `src/rag.py` retrieval before the Requirements Agent |

## Notes / known limitations

- Structured output quality depends on the local model. `llama3.2:3b` is fast but can be
  inconsistent with strict JSON; if you see parsing or validation errors, try a larger
  local model such as `llama3.1:8b` or `qwen2.5:7b`.
- Retrieval is intentionally lightweight and local. For final evaluation, `src/rag.py`
  can be upgraded to embeddings/vector search while keeping the same upload and agent
  interface.
