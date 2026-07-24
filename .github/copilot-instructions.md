# Art Heist Investigator Crew

A multi-agent AI system that investigates a fictional art heist using **CrewAI** and **Google Gemini**. Originally adapted from an SAP BTP CodeJam exercise, then reworked to run entirely locally — no SAP BTP, AI Core, or cloud platform dependency remains.

## Architecture

Three-agent sequential crew in `project/Python/starter-project/`:

| Agent | Tool | Purpose |
|---|---|---|
| Loss Appraiser | `call_rpt1()` | Predict item categories & insurance values via a local scikit-learn model (stand-in for a hosted structured-prediction service) |
| Evidence Analyst | `call_grounding_service()` | RAG queries over evidence documents via local sentence-transformer embeddings + cosine similarity (stand-in for a hosted grounding service) |
| Lead Detective | _(none)_ | Synthesise findings and name the culprit |

Key files in `project/Python/starter-project/`:
- `investigator_crew.py` — `@CrewBase` class; agents and tasks defined with `@agent`, `@task`, `@crew` decorators; tools use `@tool(...)`
- `main.py` — entry point; calls `crew().kickoff(inputs={...})`
- `rpt1_sklearn_tool.py` — local structured-prediction tool (gradient boosting), same request/response shape as the original hosted model it replaces
- `grounding_tool.py` — local RAG tool (sentence-transformers + cosine similarity)
- `payload.py` — structured art-item data with `[PREDICT]` placeholders
- `config/agents.yaml`, `config/tasks.yaml` — YAML definitions (method names must match decorator names)

Evidence documents (plain text, loaded by the grounding tool) are in `project/Python/starter-project/evidence_documents/`.

## Build & Run

```bash
cd project/Python/starter-project

# Install dependencies (only needed once)
pip install crewai python-dotenv certifi scikit-learn pandas numpy sentence-transformers

# Run the crew
python main.py
```

Requires a `.env` file in `project/Python/starter-project/`:

```
GEMINI_API_KEY=
```

## Conventions

- **CrewAI YAML config**: agent/task names in `agents.yaml` / `tasks.yaml` must exactly match the Python method names decorated with `@agent` / `@task`.
- **Tool pattern**: tools are plain functions decorated with `@tool("Descriptive Name")`. Return error messages as strings so the LLM can handle failures gracefully.
- **LLM model string**: `gemini/gemini-2.5-flash`, passed to CrewAI's `LLM(...)` with `api_key=os.getenv("GEMINI_API_KEY")`.
- **Process**: always `Process.sequential` — tasks pass outputs as context to the next task in order.
- **Structured-prediction payload**: `[PREDICT]` string is the placeholder for values to be inferred; schema (dtype, categories, value ranges) must be exact.

## Pitfalls

- **YAML / decorator name mismatch** causes silent CrewAI failures with no clear error message.
- **Grounding tool loads `evidence_documents/` once at import time** — an empty or missing folder means the Evidence Analyst has nothing to retrieve and will hallucinate.
- **No `.env` validation** at startup for the Gemini key — credential errors only surface on the first API call.
- **Corporate/enterprise networks**: `basic_agent.py` and `investigator_crew.py` set `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`/`HTTPX_SSL_VERIFY` to `certifi.where()` at import time to work around corporate TLS interception — don't remove this without checking your network setup.
