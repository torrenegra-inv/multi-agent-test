---
description: "Use when writing Python code for this CrewAI investigator crew: agent/task wiring, tool patterns, the local structured-prediction and RAG stand-ins, and Gemini LLM configuration."
applyTo: "**/*.py"
---

# Python Agent & Tools Conventions

## LLM Configuration

All agents share one Gemini `LLM` instance, configured with the `gemini/<model-name>` provider prefix and an explicit API key:

```python
from crewai import LLM

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)
```

Load `.env` (containing `GEMINI_API_KEY`) once at the top of the entry-point module, before instantiating the LLM:

```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
```

## CrewAI Agent & Task Pattern

Agents and their tasks are defined in YAML; the Python method names **must exactly match** the YAML keys:

```python
@CrewBase
class MyCrew():
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    @agent
    def appraiser_agent(self) -> Agent:   # key in agents.yaml: appraiser_agent
        return Agent(config=self.agents_config["appraiser_agent"], llm=llm, tools=[call_rpt1])

    @task
    def appraise_loss_task(self) -> Task:     # key in tasks.yaml: appraise_loss_task
        return Task(config=self.tasks_config["appraise_loss_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential)
```

Always use `Process.sequential`; task outputs are automatically passed as context to subsequent tasks (see the Lead Detective task's `context=[appraise_loss_task(), analyze_evidence_task()]`).

## Tool Pattern

Tools are module-level functions decorated with `@tool`. Return error messages as plain strings so the LLM can recover gracefully — never raise exceptions out of a tool:

```python
from crewai.tools import tool

@tool("call_rpt1")
def call_rpt1(payload: dict) -> str:
    """Docstring is the tool description shown to the agent."""
    try:
        result = _predict(payload)  # local scikit-learn model
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error calling structured-prediction tool: {str(e)}"
```

## Structured-Prediction Tool (`rpt1_sklearn_tool.py`)

A local scikit-learn `GradientBoostingRegressor`/`GradientBoostingClassifier` stands in for a hosted structured-prediction service, accepting the same payload shape:

- `"[PREDICT]"` is the placeholder string for values the model should infer.
- The `data_schema` dict (dtype, categories, numeric ranges) must match the payload rows exactly — mismatches raise during feature encoding, not silently.
- Pass the payload dict from `main.py` inputs via the crew kickoff: `crew.kickoff(inputs={"payload": payload, ...})`.

## RAG / Grounding Tool (`grounding_tool.py`)

A local stand-in for a hosted grounding service: loads all `.txt` files from `evidence_documents/` once at import time, embeds them with `sentence-transformers` (`all-MiniLM-L6-v2`), and retrieves by cosine similarity against the query.

- Documents are loaded lazily on first tool call and cached in memory — an empty or missing `evidence_documents/` folder means the tool has nothing to retrieve and the agent will hallucinate instead of citing sources.
- First call downloads the embedding model — expect a one-time delay.

## Corporate/Enterprise Networks

If outbound HTTPS calls (to Gemini or the sentence-transformers model hub) fail with SSL errors behind a corporate proxy, set these before any network call:

```python
import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["HTTPX_SSL_VERIFY"] = certifi.where()
```
