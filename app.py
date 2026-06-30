from __future__ import annotations

import datetime
import json
import re
import time
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

load_dotenv()

from agent.graph import graph
from agent.tracing import build_graph_config, get_tracing_status

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Planning Agent",
    page_icon="🗺️",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
[data-testid="stChatMessageContent"] p { margin-bottom: 0.4rem; }
.step-active  { font-weight: 700; color: #4CAF50; }
.step-done    { color: #9E9E9E; text-decoration: line-through; }
.step-pending { color: #BDBDBD; }
[data-testid="stMetric"] { text-align: center; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sessions persistence ───────────────────────────────────────────────────────
SESSIONS_FILE = Path(__file__).parent / "sessions.json"


def _load_all_sessions() -> dict:
    if SESSIONS_FILE.exists():
        try:
            with open(SESSIONS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_all_sessions(sessions: dict):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)


def _name_from_message(message: str) -> str:
    words = message.strip().split()
    return " ".join(words[:5]) if words else "Untitled"


def _save_current_session():
    tid = st.session_state.thread_id
    sessions = _load_all_sessions()
    existing = sessions.get(tid, {})
    name = existing.get("name", "")
    if not name:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                name = _name_from_message(msg["content"])
                break
    if not name:
        name = f"Session {tid[:8]}"
    sessions[tid] = {
        "name": name,
        "thread_id": tid,
        "chat_history": st.session_state.chat_history,
        "plan_draft": st.session_state.plan_draft,
        "approved": st.session_state.approved,
        "awaiting_resume": st.session_state.awaiting_resume,
        "plan_file_path": st.session_state.plan_file_path,
        "updated_at": datetime.datetime.now().isoformat(),
    }
    _save_all_sessions(sessions)


def _switch_session(thread_id: str):
    sessions = _load_all_sessions()
    s = sessions.get(thread_id)
    if s:
        st.session_state.thread_id = s["thread_id"]
        st.session_state.chat_history = s["chat_history"]
        st.session_state.plan_draft = s.get("plan_draft")
        st.session_state.approved = s.get("approved", False)
        st.session_state.awaiting_resume = s.get("awaiting_resume", False)
        st.session_state.plan_file_path = s.get("plan_file_path")
        st.session_state.session_start = time.time()
        st.session_state.exec_duration = None
        st.session_state.execution_log = []
    st.session_state.renaming_session = None
    st.rerun()


# -- Session state initialisation ----------------------------------------------
def _init_session():
    defaults = {
        "thread_id": lambda: str(uuid.uuid4()),
        "chat_history": list,
        "plan_draft": lambda: None,
        "awaiting_resume": lambda: False,
        "approved": lambda: False,
        "plan_file_path": lambda: None,
        "renaming_session": lambda: None,
        "session_start": time.time,
        "exec_duration": lambda: None,
        # Per-node execution trace accumulated across all graph runs this session.
        "execution_log": list,
    }
    for key, factory in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


_init_session()


# -- Helpers -------------------------------------------------------------------
def _get_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _sync_state_from_graph():
    """Pull plan_draft, approved, and plan_file_path out of the graph checkpoint."""
    try:
        snapshot = graph.get_state(_get_config())
        if snapshot and snapshot.values:
            plan = snapshot.values.get("plan_draft")
            if plan:
                st.session_state.plan_draft = plan
            if snapshot.values.get("approved"):
                st.session_state.approved = True
            fp = snapshot.values.get("plan_file_path")
            if fp:
                st.session_state.plan_file_path = fp
    except Exception:
        pass


def _new_visible_messages(all_messages: list, prev_count: int) -> list[str]:
    """Return content of new AIMessages (no tool calls) added after prev_count."""
    result = []
    for m in all_messages[prev_count:]:
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.content:
            result.append(m.content)
    return result


def _make_log_entry(node_name: str, updates: dict, elapsed: float) -> dict:
    """Convert a single node's state-update dict into a human-readable log entry.

    Each entry has:
        node      – node name
        elapsed   – seconds since this graph run started
        timestamp – wall-clock HH:MM:SS
        bullets   – list of short plaintext facts about what this node did
        raw_keys  – keys that changed in the state (for debugging)
    """
    bullets: list[str] = []

    # ── Messages ──────────────────────────────────────────────────────────────
    msgs = updates.get("messages", [])
    ai_with_tools = [m for m in msgs if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)]
    ai_visible    = [m for m in msgs if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.content]
    tool_results  = [m for m in msgs if not isinstance(m, AIMessage) and hasattr(m, "content")]

    for m in ai_with_tools:
        tc = getattr(m, "tool_calls", [])
        names = ", ".join(t.get("name", "?") for t in tc)
        bullets.append(f"🔧 Called tool(s): **{names}**")

    for m in tool_results:
        preview = str(m.content)[:200].replace("\n", " ")
        if len(str(m.content)) > 200:
            preview += "…"
        bullets.append(f"📥 Tool result: {preview}")

    for m in ai_visible:
        preview = m.content[:300].replace("\n", " ")
        if len(m.content) > 300:
            preview += "…"
        # Token usage if available
        usage = getattr(m, "usage_metadata", None)
        tok_str = f" · {usage['total_tokens']} tokens" if usage and usage.get("total_tokens") else ""
        bullets.append(f"💬 LLM response{tok_str}: {preview}")

    # ── Structured state fields ───────────────────────────────────────────────
    if updates.get("research_summary"):
        rs = updates["research_summary"]
        preview = rs[:300] + ("…" if len(rs) > 300 else "")
        bullets.append(f"📝 Research summary: {preview}")

    if "need_clarification" in updates:
        bullets.append(f"❓ Needs clarification: **{updates['need_clarification']}**")

    if updates.get("clarifying_questions"):
        cq = updates["clarifying_questions"]
        preview = cq[:300] + ("…" if len(cq) > 300 else "")
        bullets.append(f"🗒 Questions: {preview}")

    if updates.get("user_answers"):
        bullets.append(f"✏️ User answers: {updates['user_answers']}")

    if updates.get("plan_draft"):
        pd = updates["plan_draft"]
        bullets.append(f"📄 Plan draft produced ({len(pd)} chars)")

    if "approved" in updates:
        val = updates["approved"]
        bullets.append(f"{'✅' if val else '🔄'} Approved: **{val}**")

    if updates.get("plan_file_path"):
        bullets.append(f"💾 Saved to: `{updates['plan_file_path']}`")

    if not bullets:
        bullets.append(f"State keys updated: {list(updates.keys())}")

    return {
        "node": node_name,
        "elapsed": round(elapsed, 2),
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "bullets": bullets,
        "raw_keys": list(updates.keys()),
    }


def _run_graph(user_message: str) -> str:
    """Invoke the graph via stream() so every node's output is captured."""
    config = _get_config()

    prev_count = 0
    topic: str | None = None
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            prev_count = len(snapshot.values.get("messages", []))
    except Exception:
        pass

    if not st.session_state.chat_history:
        topic = user_message[:80]

    rich_config = build_graph_config(
        thread_id=st.session_state.thread_id,
        topic=topic,
    )

    if prev_count == 0:
        stream_input = {"messages": [HumanMessage(content=user_message)]}
    elif st.session_state.awaiting_resume:
        stream_input = Command(resume=user_message)
    else:
        stream_input = {"messages": [HumanMessage(content=user_message)]}

    run_log: list[dict] = []
    t0 = time.time()

    with st.spinner("Agent working..."):
        try:
            for chunk in graph.stream(
                stream_input,
                config=rich_config,
                stream_mode="updates",
            ):
                for node_name, node_updates in chunk.items():
                    if node_name.startswith("__"):
                        continue
                    run_log.append(_make_log_entry(node_name, node_updates, time.time() - t0))

            st.session_state.exec_duration = time.time() - t0
            st.session_state.awaiting_resume = False
            st.session_state.execution_log = st.session_state.execution_log + run_log
            _sync_state_from_graph()

            # Gather new visible messages from the final checkpoint.
            try:
                final = graph.get_state(config)
                all_msgs = final.values.get("messages", []) if final and final.values else []
                visible = _new_visible_messages(all_msgs, prev_count)
            except Exception:
                visible = []

            return "\n\n---\n\n".join(visible) if visible else "Done."

        except GraphInterrupt as exc:
            st.session_state.exec_duration = time.time() - t0
            st.session_state.awaiting_resume = True
            st.session_state.execution_log = st.session_state.execution_log + run_log
            _sync_state_from_graph()

            try:
                new_snapshot = graph.get_state(config)
                all_msgs = (
                    new_snapshot.values.get("messages", [])
                    if new_snapshot and new_snapshot.values
                    else []
                )
                visible = _new_visible_messages(all_msgs, prev_count)
            except Exception:
                visible = []

            try:
                interrupt_val = str(exc.args[0][0].value)
            except (IndexError, AttributeError, TypeError):
                interrupt_val = str(exc.args[0]) if exc.args else ""

            if interrupt_val and interrupt_val not in visible:
                visible.append(interrupt_val)

            return "\n\n---\n\n".join(visible) if visible else interrupt_val

        except Exception as exc:
            st.session_state.exec_duration = time.time() - t0
            st.session_state.execution_log = st.session_state.execution_log + run_log
            st.session_state.awaiting_resume = False
            return f"⚠️ Error: {exc}"


def _new_session():
    for key in [
        "thread_id", "chat_history", "plan_draft", "awaiting_resume",
        "approved", "plan_file_path", "exec_duration", "session_start",
        "execution_log",
    ]:
        st.session_state.pop(key, None)
    st.session_state.renaming_session = None
    _init_session()
    st.rerun()


# ── Workflow phases ────────────────────────────────────────────────────────────
_PHASES: list[tuple[str, str, str]] = [
    ("ready",            "Ready",            "⬜"),
    ("discovery",        "Discovery",        "🔍"),
    ("research_summary", "Research Summary", "📝"),
    ("alignment",        "Alignment",        "🤝"),
    ("design",           "Design",           "🎨"),
    ("review",           "Review",           "👀"),
    ("complete",         "Complete",         "✅"),
]
_PHASE_ORDER = [p[0] for p in _PHASES]
_PHASE_LABEL = {p[0]: p[1] for p in _PHASES}
_PHASE_ICON  = {p[0]: p[2] for p in _PHASES}


def _get_phase() -> str:
    """Determine the current workflow phase from session state and graph state."""
    if st.session_state.approved:
        return "complete"
    if st.session_state.plan_draft:
        return "review"
    try:
        snapshot = graph.get_state(_get_config())
        if snapshot and snapshot.values:
            vals = snapshot.values
            if vals.get("plan_draft"):
                return "review"
            if vals.get("research_summary"):
                next_nodes = list(snapshot.next or [])
                if any("design" in n for n in next_nodes):
                    return "design"
                if any("alignment" in n for n in next_nodes):
                    return "alignment"
                return "research_summary"
            if vals.get("messages"):
                return "discovery"
    except Exception:
        pass
    if st.session_state.chat_history:
        return "discovery"
    return "ready"


def _count_metrics() -> dict:
    """Count LLM calls, tool calls, and tokens from the graph checkpoint."""
    m: dict = {"llm_calls": 0, "tool_calls": 0, "sources": 0, "tokens": 0}
    try:
        snapshot = graph.get_state(_get_config())
        if not snapshot or not snapshot.values:
            return m
        for msg in snapshot.values.get("messages", []):
            if not isinstance(msg, AIMessage):
                continue
            tc = getattr(msg, "tool_calls", None) or []
            if tc:
                m["tool_calls"] += len(tc)
                m["sources"] += len(tc)
            elif msg.content:
                m["llm_calls"] += 1
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                m["tokens"] += usage.get("total_tokens", 0)
    except Exception:
        pass
    return m


def _parse_plan_sections(md: str) -> dict[str, str]:
    """Split a plan markdown into named sections using header anchors."""
    sections: dict[str, str] = {}
    patterns = [
        ("tldr",           r"###\s*TL;DR\s*\n(.*?)(?=\n###|\Z)"),
        ("steps",          r"###\s*Steps\s*\n(.*?)(?=\n###|\Z)"),
        ("references",     r"###\s*Relevant Files[^\n]*\n(.*?)(?=\n###|\Z)"),
        ("verification",   r"###\s*Verification\s*\n(.*?)(?=\n###|\Z)"),
        ("decisions",      r"###\s*Decisions\s*\n(.*?)(?=\n###|\Z)"),
        ("exclusions",     r"###\s*Scope Exclusions\s*\n(.*?)(?=\n###|\Z)"),
        ("considerations", r"###\s*Further Considerations\s*\n(.*?)(?=\n###|\Z)"),
    ]
    for key, pattern in patterns:
        m = re.search(pattern, md, re.DOTALL | re.IGNORECASE)
        if m:
            sections[key] = m.group(1).strip()
    return sections


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
current_phase = _get_phase()

with st.sidebar:
    st.title("🗺️ Planning Agent")
    st.caption("AI-powered multi-phase planning workspace.")

    # ── Workflow stepper ──────────────────────────────────────────────────────
    st.markdown("### Workflow")
    current_idx = _PHASE_ORDER.index(current_phase)
    for _pk, _pl, _pi in _PHASES[1:]:   # skip "ready"
        _pidx = _PHASE_ORDER.index(_pk)
        if _pk == current_phase:
            st.markdown(f"**→ {_pi} {_pl}**")
        elif _pidx < current_idx:
            st.markdown(
                f"<span class='step-done'>{_pi} {_pl}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<span class='step-pending'>{_pi} {_pl}</span>",
                unsafe_allow_html=True,
            )

    st.divider()

    if st.button("➕ New Session", use_container_width=True):
        _new_session()

    st.divider()

    # ── LangSmith status ──────────────────────────────────────────────────────
    _tracing = get_tracing_status()
    if _tracing["active"]:
        st.success(f"🔍 LangSmith · **{_tracing['project']}**")
    else:
        st.caption(
            "🔍 LangSmith inactive — set `LANGCHAIN_TRACING_V2=true` "
            "and `LANGCHAIN_API_KEY` in your `.env`."
        )

    st.divider()

    # ── Session history ───────────────────────────────────────────────────────
    st.subheader("Sessions")
    all_sessions = _load_all_sessions()
    sorted_sessions = sorted(
        all_sessions.values(),
        key=lambda s: s.get("updated_at", ""),
        reverse=True,
    )

    if not sorted_sessions:
        st.caption("No saved sessions yet.")
    else:
        for s in sorted_sessions:
            tid = s["thread_id"]
            name = s.get("name") or f"Session {tid[:8]}"
            is_active = tid == st.session_state.thread_id

            if st.session_state.renaming_session == tid:
                new_name = st.text_input(
                    "New name",
                    value=name,
                    key=f"rename_input_{tid}",
                    label_visibility="collapsed",
                )
                save_col, cancel_col = st.columns(2)
                with save_col:
                    if st.button("✔ Save", key=f"save_rename_{tid}", use_container_width=True):
                        sessions = _load_all_sessions()
                        if tid in sessions:
                            sessions[tid]["name"] = new_name.strip() or name
                            _save_all_sessions(sessions)
                        st.session_state.renaming_session = None
                        st.rerun()
                with cancel_col:
                    if st.button("✖ Cancel", key=f"cancel_rename_{tid}", use_container_width=True):
                        st.session_state.renaming_session = None
                        st.rerun()
            else:
                label_str = f"**› {name}**" if is_active else name
                btn_col, edit_col, del_col = st.columns([5, 1, 1])
                with btn_col:
                    if st.button(label_str, key=f"session_{tid}", use_container_width=True):
                        if not is_active:
                            _switch_session(tid)
                with edit_col:
                    if st.button("✏️", key=f"rename_{tid}", help="Rename", use_container_width=True):
                        st.session_state.renaming_session = tid
                        st.rerun()
                with del_col:
                    if st.button("🗑️", key=f"delete_{tid}", help="Delete", use_container_width=True):
                        sessions = _load_all_sessions()
                        sessions.pop(tid, None)
                        _save_all_sessions(sessions)
                        if is_active:
                            _new_session()
                        else:
                            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
h_left, h_mid, h_right = st.columns([3, 3, 2])
with h_left:
    st.markdown("## 🗺️ Planning Agent")
with h_mid:
    _phase_label = _PHASE_LABEL.get(current_phase, current_phase.title())
    _phase_icon  = _PHASE_ICON.get(current_phase, "")
    st.markdown(f"**Phase:** {_phase_icon} {_phase_label}")
with h_right:
    _elapsed_s = int(time.time() - st.session_state.session_start)
    _elapsed_str = f"{_elapsed_s // 60}m {_elapsed_s % 60}s"
    st.markdown(f"**Session:** `{st.session_state.thread_id[:8]}` · ⏱ {_elapsed_str}")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT  —  Chat  |  Plan Workspace
# ══════════════════════════════════════════════════════════════════════════════
chat_col, plan_col = st.columns([5, 5])

# ── LEFT: Chat ─────────────────────────────────────────────────────────────────
with chat_col:
    st.markdown("### 💬 Chat")

    for msg in st.session_state.chat_history:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if st.session_state.approved:
        st.info("✅ Planning complete. Start a new session to plan something else.")
    else:
        placeholder = (
            "Reply to the agent..."
            if st.session_state.awaiting_resume
            else "Describe what you want to plan..."
        )
        user_input = st.chat_input(placeholder)

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)

            agent_response = _run_graph(user_input)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": agent_response}
            )
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(agent_response)

            _save_current_session()
            _sync_state_from_graph()
            st.rerun()

# ── RIGHT: Plan Workspace ──────────────────────────────────────────────────────
with plan_col:

    if st.session_state.approved:
        # ── Completion screen ────────────────────────────────────────────────
        st.markdown("### 🎉 Planning Complete")
        st.success("Your plan has been approved and saved.")
        st.markdown("---")

        if st.session_state.plan_draft:
            _file_name = (
                Path(st.session_state.plan_file_path).name
                if st.session_state.plan_file_path
                else "plan.md"
            )
            dl_col, new_col = st.columns(2)
            with dl_col:
                st.download_button(
                    label="⬇️ Download Markdown",
                    data=st.session_state.plan_draft.encode("utf-8"),
                    file_name=_file_name,
                    mime="text/markdown",
                    use_container_width=True,
                    type="primary",
                )
            with new_col:
                if st.button("🔄 Start New Session", use_container_width=True):
                    _new_session()

    elif st.session_state.plan_draft:
        # ── Plan workspace with tabs ─────────────────────────────────────────
        st.markdown("### 📋 Plan Workspace")

        _sections = _parse_plan_sections(st.session_state.plan_draft)

        tab_overview, tab_plan, tab_refs, tab_verify, tab_decisions = st.tabs(
            ["Overview", "Plan", "References", "Verification", "Decisions"]
        )

        with tab_overview:
            _title_m = re.search(r"##\s*Plan:\s*(.*)", st.session_state.plan_draft)
            if _title_m:
                st.markdown(f"# {_title_m.group(1).strip()}")
            if _sections.get("tldr"):
                st.markdown(_sections["tldr"])
            if _sections.get("considerations"):
                st.markdown("#### Further Considerations")
                st.markdown(_sections["considerations"])
            with st.expander("📋 Copy raw (full plan)"):
                st.code(st.session_state.plan_draft, language="markdown")

        with tab_plan:
            if _sections.get("steps"):
                st.markdown(_sections["steps"])
            else:
                st.markdown(st.session_state.plan_draft)
            with st.expander("📋 Copy raw"):
                st.code(_sections.get("steps", ""), language="markdown")

        with tab_refs:
            if _sections.get("references"):
                st.markdown(_sections["references"])
            else:
                st.info("No references section found in this plan.")
            with st.expander("📋 Copy raw"):
                st.code(_sections.get("references", ""), language="markdown")

        with tab_verify:
            if _sections.get("verification"):
                st.markdown(_sections["verification"])
            else:
                st.info("No verification section found in this plan.")
            if _sections.get("exclusions"):
                st.markdown("#### Scope Exclusions")
                st.markdown(_sections["exclusions"])

        with tab_decisions:
            if _sections.get("decisions"):
                st.markdown(_sections["decisions"])
            else:
                st.info("No decisions section found in this plan.")

        st.divider()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("✅ Approve Plan", use_container_width=True, type="primary"):
                approval_response = _run_graph("Approved")
                st.session_state.chat_history.append({"role": "user", "content": "Approved"})
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": approval_response}
                )
                _save_current_session()
                _sync_state_from_graph()
                st.rerun()

        with btn_col2:
            if st.button("✏️ Request Changes", use_container_width=True):
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": "What changes would you like to make to the plan?",
                    }
                )
                st.rerun()

    else:
        # ── Pre-plan placeholder ─────────────────────────────────────────────
        st.markdown("### 📋 Plan Workspace")
        if st.session_state.awaiting_resume:
            st.info(
                f"⏳ Waiting for your input in the "
                f"**{_PHASE_LABEL.get(current_phase, current_phase)}** phase."
            )
        else:
            st.info("The plan will appear here once the Design phase is complete.")


# ══════════════════════════════════════════════════════════════════════════════
# METRICS BAR
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("#### 📊 Session Metrics")

_metrics = _count_metrics()
_total_elapsed = int(time.time() - st.session_state.session_start)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("LLM Calls", _metrics.get("llm_calls", 0))
with m2:
    st.metric("Tool Calls", _metrics.get("tool_calls", 0))
with m3:
    st.metric("Sources", _metrics.get("sources", 0))
with m4:
    _t_str = f"{_total_elapsed // 60}m {_total_elapsed % 60}s"
    _last_exec = st.session_state.exec_duration
    _delta = f"last: {_last_exec:.1f}s" if _last_exec else None
    st.metric("Session Time", _t_str, delta=_delta)
with m5:
    _tok = _metrics.get("tokens", 0)
    st.metric("Tokens", f"{_tok:,}" if _tok else "—")


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION TRACE  —  Inspector
# Every node that ran this session is shown here with its key outputs.
# ══════════════════════════════════════════════════════════════════════════════
_log = st.session_state.execution_log

# Map node name → phase label + icon for a richer header.
_NODE_META: dict[str, tuple[str, str]] = {
    "discovery":        ("🔍", "Discovery"),
    "tool_executor":    ("🔧", "Tool Executor"),
    "research_summary": ("📝", "Research Summary"),
    "alignment":        ("🤝", "Alignment"),
    "design":           ("🎨", "Design"),
    "present_plan":     ("👀", "Present Plan"),
    "refinement":       ("✏️", "Refinement"),
}

if _log:
    with st.expander(
        f"🔬 Execution Trace — {len(_log)} node run(s)  ·  click to expand",
        expanded=False,
    ):
        # Clear button
        if st.button("🗑 Clear trace", key="clear_trace"):
            st.session_state.execution_log = []
            st.rerun()

        for i, entry in enumerate(_log):
            node = entry["node"]
            icon, label = _NODE_META.get(node, ("▶", node))
            elapsed = entry.get("elapsed", "?")
            ts = entry.get("timestamp", "")

            # Node header row
            st.markdown(
                f"**{i + 1}. {icon} {label}** "
                f"<span style='color:#888;font-size:0.82rem'>@ {ts} · +{elapsed}s</span>",
                unsafe_allow_html=True,
            )

            # Bullet-point summary
            for bullet in entry.get("bullets", []):
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{bullet}", unsafe_allow_html=True)

            # Raw keys changed (collapsed)
            raw_keys = entry.get("raw_keys", [])
            if raw_keys:
                st.caption(f"State keys updated: `{'`, `'.join(raw_keys)}`")

            if i < len(_log) - 1:
                st.markdown("---")
else:
    with st.expander("🔬 Execution Trace — no runs yet", expanded=False):
        st.caption(
            "Node-by-node outputs will appear here after the first graph run. "
            "Each entry shows which tools were called, what the LLM responded, "
            "and how state changed at every step."
        )

