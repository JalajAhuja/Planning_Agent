# 🗺️ Planning Agent

> A multi-phase AI planning workspace powered by **LangGraph**, **GPT-4o / GPT-5**, and **Streamlit**.  
> Tell it what you want to build — it researches, asks exactly the right questions, and produces a structured, downloadable plan.

---

## Table of Contents

- [What it does](#what-it-does)
- [How it works — the five phases](#how-it-works--the-five-phases)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Environment variables](#environment-variables)
- [UI tour](#ui-tour)
- [LangSmith tracing](#langsmith-tracing)
- [Session management](#session-management)
- [Plan output format](#plan-output-format)
- [Extending the agent](#extending-the-agent)

---

## What it does

You describe a goal — *"Build a REST API for a task manager"*, *"Migrate our Postgres schema to multi-tenant"*, *"Design a CI/CD pipeline for a monorepo"* — and the agent:

1. **Researches** your topic autonomously using web search and Wikipedia.
2. **Asks targeted questions** when it genuinely needs clarification (max 4 questions, never redundant ones).
3. **Produces a structured plan** with phases, concrete steps, file/command references, verification checks, and design decisions.
4. **Iterates** on the plan based on your feedback until you approve it.
5. **Saves and exports** the final plan as a Markdown file you can download directly from the UI.

Every run is observable through a live **Execution Trace** panel and optionally in **LangSmith** for deep nested traces.

---

## How it works — the five phases

```
User message
     │
     ▼
┌──────────────┐     tool calls      ┌───────────────┐
│  Discovery   │ ─────────────────►  │ Tool Executor │
│  (GPT-4o)    │ ◄────────────────── │ (web/wiki/fs) │
└──────┬───────┘   results looped    └───────────────┘
       │ done researching
       ▼
┌──────────────────┐
│ Research Summary │  synthesises all tool results → RouterDecision
│  (GPT-4o)        │  sets: research_summary, need_clarification
└──────┬───────────┘
       │
       ▼
┌────────────┐   need_clarification=True   interrupt()
│ Alignment  │ ──────────────────────────► waits for answers
│ (GPT-4o)   │ ◄────────────────────────── user replies
└──────┬─────┘
       │
       ▼
┌────────────┐
│   Design   │  produces complete structured plan
│ (GPT-5)    │
└──────┬─────┘
       │
       ▼
┌──────────────┐   interrupt()
│ Present Plan │ ──────────────► shows plan in UI, waits for feedback
│              │ ◄────────────── user: approve / revise / question
└──────┬───────┘
       │
       ▼
┌────────────┐   approved  ──► saves Markdown → END
│ Refinement │   revision  ──► updates plan_draft → Present Plan
│ (GPT-4o)   │   question  ──► answers inline → Present Plan
│            │   alternative ► clears state → Discovery
└────────────┘
```

### Phase details

| Phase | Node(s) | Model | Purpose |
|---|---|---|---|
| **Discovery** | `discovery` + `tool_executor` | GPT-4o | ReAct loop — calls Tavily Search, Wikipedia, and local file reader until research is complete. Never summarises or talks to the user. |
| **Research Summary** | `research_summary` | GPT-4o | Reads every tool result and emits a `RouterDecision`: structured research summary + clarification flag. |
| **Alignment** | `alignment` | GPT-4o | Identifies the minimal questions needed; asks them in one round via `interrupt()`. Skips entirely if research is sufficient. |
| **Design** | `design` | GPT-5 | Writes the full plan using a strict template (TL;DR, Steps, References, Verification, Decisions, Exclusions, Further Considerations). |
| **Review / Refinement** | `present_plan` + `refinement` | GPT-4o | Presents the plan, waits for user feedback, classifies it (approval / revision / question / alternative), and acts accordingly. |

---

## Architecture

```
agent/
├── graph.py        — StateGraph definition, node wiring, SQLite checkpointer
├── nodes.py        — Node functions (discovery, research_summary, alignment,
│                     design, present_plan, refinement)
├── state.py        — PlannerState TypedDict + Pydantic output schemas
├── llms.py         — LLM instances with run names & LangSmith tags
├── prompts.py      — System prompts for every node
├── routers.py      — Conditional edge functions
├── tools.py        — Tavily Search, Wikipedia, local file reader
├── formatter.py    — save_plan(), PLAN_TEMPLATE, format_plan()
└── tracing.py      — LangSmith helpers: build_graph_config(), get_tracing_status()

app.py              — Streamlit UI (chat, plan workspace, execution trace, metrics)
main.py             — Entry point: launches `streamlit run app.py`
pyproject.toml      — Dependencies & project metadata
sessions.json       — Persisted session history (auto-managed)
checkpoints.db      — SQLite LangGraph checkpoint store (auto-managed)
plans/              — Saved Markdown plans (created on first approval)
```

### State schema (`PlannerState`)

| Field | Type | Set by |
|---|---|---|
| `messages` | `list[BaseMessage]` | All nodes (append-only via `add_messages`) |
| `research_summary` | `str` | `research_summary` |
| `need_clarification` | `bool` | `research_summary` |
| `clarifying_questions` | `str` | `alignment` |
| `user_answers` | `list[str]` | `alignment` (via interrupt) |
| `plan_draft` | `str \| None` | `design`, `refinement` |
| `approved` | `bool` | `refinement` |
| `final_plan` | `str \| None` | `refinement` |
| `plan_file_path` | `str \| None` | `refinement` (on approval) |

---

## Project structure

```
Planning Agent/
├── agent/
│   ├── __init__.py
│   ├── formatter.py
│   ├── graph.py
│   ├── llms.py
│   ├── nodes.py
│   ├── prompts.py
│   ├── routers.py
│   ├── state.py
│   ├── tools.py
│   └── tracing.py
├── .github/
│   └── Planning_Agent_Enhancement_Spec.md
├── app.py
├── main.py
├── pyproject.toml
├── sessions.json       ← auto-created
├── checkpoints.db      ← auto-created
└── plans/              ← auto-created on first approval
```

---

## Setup

### Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) (recommended) **or** `pip`
- OpenAI API key
- Tavily API key (free tier at [tavily.com](https://tavily.com))
- *(optional)* LangSmith API key for tracing

### Install

```bash
# Clone the repo
git clone <your-repo-url>
cd "Planning Agent"

# Install dependencies with uv (creates .venv automatically)
uv pip install -r pyproject.toml

# Or with pip
pip install -e .
```

### Configure environment

Create a `.env` file in the project root:

```dotenv
# Required
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Optional — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=planning-agent   # project name in LangSmith dashboard

# Optional — restrict local file reader to a specific workspace
WORKSPACE_ROOT=.
```

---

## Running the app

```bash
# Recommended — uses the pyproject.toml entry point
uv run python main.py

# Or directly
uv run streamlit run app.py

# Or with plain Python (if .venv is activated)
python main.py
```

The app opens at **http://localhost:8501** by default.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `TAVILY_API_KEY` | ✅ | — | Tavily web search key |
| `LANGCHAIN_TRACING_V2` | ☐ | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | ☐ | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | ☐ | `planning-agent` | LangSmith project name |
| `LANGCHAIN_ENDPOINT` | ☐ | `https://api.smith.langchain.com` | LangSmith endpoint |
| `WORKSPACE_ROOT` | ☐ | `.` | Root directory for the `read_file_tool` |

---

## UI tour

### Header
Persistent bar showing the app title, current workflow phase (e.g. `🔍 Discovery`), the abbreviated session ID, and a live elapsed timer.

### Sidebar
- **Workflow stepper** — visual progress indicator through Discovery → Research Summary → Alignment → Design → Review → Complete. Active phase is bolded; completed phases are struck through.
- **New Session** button — starts a fresh plan in a new thread.
- **LangSmith status** — green badge when tracing is active, grey hint otherwise.
- **Session history** — list of all past sessions with rename (✏️) and delete (🗑️) controls. Click any session to restore it.

### Chat (left column)
Standard chat interface. User and assistant messages use distinct avatars (🧑 / 🤖). The input is disabled after plan approval.

### Plan Workspace (right column)
Appears once the Design phase completes. Five tabs:

| Tab | Content |
|---|---|
| **Overview** | Plan title + TL;DR + Further Considerations |
| **Plan** | All steps grouped by phase |
| **References** | Relevant files, libraries, and commands |
| **Verification** | Checklist to confirm each phase works |
| **Decisions** | Design decisions and scope exclusions |

Each tab has a `📋 Copy raw` expander showing the raw Markdown for easy copying.

Below the tabs: **✅ Approve Plan** and **✏️ Request Changes** buttons.

### Completion screen
Replaces the Plan Workspace after approval:
- **⬇️ Download Markdown** — saves the plan as a `.md` file (via `st.download_button`).
- **🔄 Start New Session** — resets everything.

### Session Metrics bar
Five `st.metric` tiles at the bottom of every page:

| Metric | Source |
|---|---|
| LLM Calls | Count of non-tool AIMessages in checkpoint |
| Tool Calls | Count of tool-call AIMessages |
| Sources | Number of individual tool invocations |
| Session Time | Wall-clock since session started; delta = last run duration |
| Tokens | `usage_metadata.total_tokens` summed across all AIMessages |

### Execution Trace panel
Collapsible `🔬 Execution Trace` expander at the bottom. Shows every node that ran, in order, with:
- Node name, phase icon, wall-clock timestamp, elapsed seconds
- Bullet-point summary: tools called, tool results (preview), LLM responses (preview + token count), state field changes
- `State keys updated` caption showing exactly which fields changed

---

## LangSmith tracing

When `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set, every run is sent to LangSmith with full nested traces:

```
planning-agent  (graph run)
├── discovery
│   ├── discovery-tools  (LLM — GPT-4o)
│   │   └── tavily_search (tool)
│   │   └── wikipedia     (tool)
│   └── discovery-tools  (LLM — next iteration)
├── research_summary
│   └── discovery-summary (LLM — GPT-4o)
├── alignment
│   └── alignment         (LLM — GPT-4o)
├── design
│   └── design            (LLM — GPT-5)
├── present_plan
└── refinement
    └── refinement        (LLM — GPT-4o)
```

Each run carries metadata: `thread_id`, `topic`, `timestamp`, `source: planning-agent`.  
Tags on every run: `planning-agent`, `streamlit`, plus per-LLM tags like `discovery`, `alignment`, `design`.

Interrupts across multiple user turns are all linked by the same `thread_id`, so you can follow a full planning session as one continuous trace.

---

## Session management

Sessions are persisted in `sessions.json`. Each entry stores:

```json
{
  "thread_id": "uuid",
  "name": "Build a REST API for...",
  "chat_history": [...],
  "plan_draft": "## Plan: ...",
  "approved": false,
  "awaiting_resume": true,
  "plan_file_path": "plans/plan_20260630_142301.md",
  "updated_at": "2026-06-30T14:23:01.123456"
}
```

The LangGraph checkpoint (tool call history, intermediate state) is stored in `checkpoints.db` keyed by `thread_id`, so you can switch sessions and resume mid-flow.

---

## Plan output format

Every generated plan follows this exact structure:

```markdown
## Plan: {Concise Title}

### TL;DR
2–3 sentences: what is being built, the core approach, the primary constraint.

### Steps

**Phase 1 — {Name}**
1. Step — `file_or_command` *(parallel)*
2. Step

**Phase 2 — {Name}**
3. Step

### Relevant Files & References
- `path/to/file.py` — reason it is relevant

### Verification
1. Command or check to confirm Phase 1 is complete
2. Command or check to confirm Phase 2 is complete

### Decisions
- **Decision title**: Chose X over Y because reason.

### Scope Exclusions
- What was deliberately left out

### Further Considerations
1. At most 3 items genuinely worth noting but out of current scope
```

Approved plans are saved to `plans/plan_YYYYMMDD_HHMMSS.md` and are immediately downloadable from the completion screen.

---

## Extending the agent

### Add a new tool

Edit `agent/tools.py`:

```python
@tool
def my_new_tool(query: str) -> str:
    """Description of what this tool does."""
    ...

tools: list = [tavily_search, wikipedia, read_file_tool, my_new_tool]
```

The tool is automatically available in the Discovery ReAct loop.

### Change the LLM model

Edit the constants at the top of `agent/llms.py`:

```python
FAST_MODEL  = "gpt-4o"    # used by discovery, alignment, refinement
SMART_MODEL = "gpt-5.2"   # used by design (the main plan writer)
```

### Add a new graph node

1. Write the node function in `agent/nodes.py`.
2. Import and register it in `agent/graph.py` with `builder.add_node(...)`.
3. Wire edges with `builder.add_edge(...)` or `builder.add_conditional_edges(...)`.
4. Add a matching entry in `_NODE_META` in `app.py` so the Execution Trace displays a friendly label.

### Adjust prompt behaviour

All system prompts live in `agent/prompts.py`. Each prompt is a plain string injected as a `SystemMessage`.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `langgraph` | ≥ 0.2 | Graph execution engine, interrupt/resume, checkpointing |
| `langchain-openai` | ≥ 0.3 | GPT-4o / GPT-5 LLM wrappers |
| `langchain-community` | ≥ 0.3 | Tavily Search, Wikipedia tools |
| `langsmith` | ≥ 0.2 | Distributed tracing and run observability |
| `streamlit` | ≥ 1.35 | Web UI |
| `langgraph-checkpoint-sqlite` | ≥ 3.1 | SQLite-backed state persistence |
| `python-dotenv` | ≥ 1.0 | `.env` file loading |
| `pydantic` | ≥ 2.0 | Structured LLM output schemas |

---

## License

MIT
 