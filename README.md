# Art Heist Investigator Crew

A multi-agent [CrewAI](https://github.com/crewAIInc/crewAI) system that investigates a fictional art heist using three specialized AI agents. Originally adapted from an SAP BTP CodeJam exercise, then reworked to run **entirely locally** on Google Gemini — no SAP BTP, AI Core, or cloud platform required.

## The scenario

A masterpiece has been stolen from a city's most secure gallery. No forced entry. Three suspects: the night manager on duty, a recently fired security technician, and a shadowy fence. Instead of solving the case by hand, this project builds a crew of AI agents that build the case themselves — from structured data and documented evidence, not guesswork.

## Architecture

Three agents run in sequence, each restricted to exactly one authoritative source — none of them are allowed to freelance an answer:

| Agent | Tool | Rule |
|---|---|---|
| **Loss Appraiser** | `call_rpt1` — a local scikit-learn model standing in for a hosted structured-prediction service | Never estimates a value itself |
| **Evidence Analyst** | `call_grounding_service` — local sentence-transformer embeddings + cosine similarity, standing in for a hosted RAG/grounding service | Never fabricates a fact |
| **Lead Detective** | *(none)* | Only synthesizes the other two agents' outputs into a cited final verdict |

All three agents share one Gemini (`gemini-2.5-flash`) LLM instance.

## Project structure

```
project/Python/starter-project/
  main.py                 ← entrypoint: runs the full 3-agent crew
  basic_agent.py          ← earlier, simpler single-agent (Loss Appraiser only) version
  investigator_crew.py    ← @CrewBase class wiring all three agents/tasks together
  config/agents.yaml      ← agent definitions (role, goal, backstory)
  config/tasks.yaml       ← task definitions
  rpt1_sklearn_tool.py    ← local stand-in for structured prediction (item value/category)
  grounding_tool.py       ← local stand-in for RAG grounding over evidence documents
  evidence_documents/     ← evidence text files the grounding tool searches
  payload.py              ← structured input data (the stolen items) for the appraiser
```

## Setup

From `project/Python/starter-project/`:

```bash
pip install "crewai[google-genai]" python-dotenv certifi scikit-learn pandas numpy sentence-transformers
```

Note the `[google-genai]` extra — plain `pip install crewai` does **not** include Gemini support and will fail once an agent tries to call the LLM.

If you're on a corporate network with a TLS-inspecting proxy, also install (see [Troubleshooting](#troubleshooting)):

```bash
pip install pip-system-certs
```

### Using `uv` instead

```bash
uv init
uv add "crewai[google-genai]" python-dotenv certifi scikit-learn pandas numpy sentence-transformers pip-system-certs
```

Create a `.env` file in that same folder with:

```
keep double quotes in both API key and llm
GEMINI_API_KEY="your-api-key-here"
llm="gemini/gemini-2.5-flash"
```

## Run

```bash
cd project/Python/starter-project
python main.py
# or, if you used uv:
uv run python main.py
```

This runs the full crew and prints the final investigation verdict, with every claim traced back to a specific evidence file or predicted value.

To run just the single-agent version from the earlier build step instead:

```bash
python basic_agent.py
```

Note: the first run downloads the `all-MiniLM-L6-v2` sentence-transformers model, so expect a one-time delay.

## Troubleshooting

**`SSL: CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`**

This means something between you and Gemini/Hugging Face — almost always a corporate network's TLS-inspecting proxy — is re-signing HTTPS traffic with an internal root CA that Python's default certificate bundle (`certifi`) doesn't trust. Fix:

```bash
pip install pip-system-certs
# if this project uses uv instead of a plain pip venv:
uv add pip-system-certs
```

This patches Python's SSL handling to trust your OS's certificate store (where IT installs that internal root CA), with no code changes needed. **Make sure you install it into the exact environment that runs `main.py`** — installing it into a different Python/venv than the one you actually run has no effect (this is the most common way this fix silently "doesn't work").

**`429 RESOURCE_EXHAUSTED` / quota exceeded**

Your Gemini API key has hit its rate limit — the free tier caps `gemini-2.5-flash` at a small number of requests per day. Wait for the quota to reset (the error message includes a retry delay) or use a key with a higher-tier plan. See [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

## License

Apache 2.0 — see [LICENSE](LICENSE).
