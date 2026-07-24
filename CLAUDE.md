# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **multi-agent CrewAI investigation crew** — an art-heist scenario solved by three agents (Loss Appraiser, Evidence Analyst, Lead Detective). Originally adapted from an SAP BTP CodeJam exercise, but reworked to run **entirely locally**: Gemini as the LLM, a local scikit-learn model standing in for a hosted structured-prediction service, and local sentence-transformer embeddings standing in for a hosted RAG/grounding service. No SAP BTP, AI Core, or Cloud Foundry dependency remains.

## Repo layout

```
project/Python/starter-project/   ← the actual project — everything lives here
  main.py                          ← entrypoint: InvestigatorCrew().crew().kickoff(...)
  basic_agent.py                   ← earlier single-agent version (Loss Appraiser only)
  investigator_crew.py             ← @CrewBase class wiring all three agents/tasks
  config/agents.yaml, config/tasks.yaml  ← agent/task definitions (CrewAI config-driven pattern)
  rpt1_sklearn_tool.py             ← local stand-in for SAP-RPT-1 structured prediction
  grounding_tool.py                ← local stand-in for SAP Grounding Service (RAG)
  evidence_documents/               ← evidence text files the grounding tool embeds/searches
  payload.py                        ← structured input data (stolen items) for the appraiser
presentations/                     ← slide/talk material about this project
```

## Run commands

From `project/Python/starter-project/`:

```bash
pip install "crewai[google-genai]" python-dotenv certifi scikit-learn pandas numpy sentence-transformers
python main.py
```

`.env` in that same folder must define `GEMINI_API_KEY`. First run downloads the `all-MiniLM-L6-v2` sentence-transformers model (one-time delay).

## Architecture

Three sequential agents, each locked to exactly one authoritative source — no agent is allowed to freelance an answer:

| Agent | Tool | Rule |
|---|---|---|
| `appraiser_agent` | `call_rpt1` (local sklearn model) | Never estimates a value itself |
| `evidence_analyst_agent` | `call_grounding_service` (local embeddings + cosine similarity) | Never fabricates a fact |
| `lead_detective_agent` | _(none)_ | Only synthesizes the other two agents' task outputs (`context=[...]`) into a cited verdict |

- `investigator_crew.py` is a `@CrewBase` class; agents/tasks defined via `@agent`/`@task`/`@crew` decorators.
- **Method names must exactly match the keys in `config/agents.yaml` and `config/tasks.yaml`.** Mismatches fail silently with no clear error.
- Tools are plain functions decorated `@tool(...)`. They return error strings (not raise) so the LLM can recover.
- Always `Process.sequential` — task outputs flow as context to the next task in declared order.
- All agents share one `gemini_llm = LLM(model="gemini/gemini-2.5-flash", ...)` instance.
- RPT-1-shaped payloads (`payload.py`) use `[PREDICT]` as the inference placeholder; the sklearn tool mimics RPT-1's request/response schema so the rest of the code didn't need to change when swapping in the local model.

## Known pitfalls

- **YAML/decorator name mismatch** in CrewAI is the #1 silent failure mode — check both `investigator_crew.py` and `config/*.yaml` when an agent or task "just doesn't run."
- Grounding tool loads and embeds `evidence_documents/` once at import time — an empty or missing folder means the Evidence Analyst has nothing to retrieve and will hallucinate.
- Corporate/enterprise environments may need the `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`/`HTTPX_SSL_VERIFY` env vars set to `certifi.where()` (already done at the top of `basic_agent.py` and `investigator_crew.py`) for outbound HTTPS calls to succeed.
- **These env vars are not sufficient for `httpx`/`google-genai`** (the library `crewai`'s Gemini integration uses) — `httpx` builds its default SSL context from `certifi.where()` directly and ignores those env vars. On networks with a TLS-inspecting corporate proxy, the actual fix is `pip install pip-system-certs` (or `uv add pip-system-certs` in a `uv` project), installed into the exact environment that runs the code — installing it into a different Python/venv silently does nothing.
- `429 RESOURCE_EXHAUSTED` from Gemini is a free-tier daily quota limit (small request cap on `gemini-2.5-flash`), not a bug — wait for reset or use a higher-tier key.

## What CLAUDE should not do here

- Don't reintroduce SAP BTP/AI Core/Cloud Foundry dependencies — the whole point of this adaptation is that it runs locally with no cloud platform.
- Don't add tests, linters, or CI unless explicitly requested.
- Don't "modernize" the tool implementations (sklearn model, local embeddings) into hosted services unless that's the explicit ask — they're intentional stand-ins that keep the rest of the code unchanged.
