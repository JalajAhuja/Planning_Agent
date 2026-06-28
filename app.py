import uuid
import json
import datetime
from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from dotenv import load_dotenv
load_dotenv()
from agent.graph import graph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Planning Agent",
    page_icon="🗺️",
    layout="wide",
)

# ── Phase constants ───────────────────────────────────────────────────────────
# graph.py writes lowercase phase strings into state.
# app.py displays Title-case labels. _normalise_phase() bridges the two.
PHASE_ORDER = ["discovery", "alignment", "design", "refinement", "complete"]
PHASE_LABELS = {
    "discovery":  "Discovery",
    "alignment":  "Alignment",
    "design":     "Design",
    "refinement": "Refinement",
    "complete":   "Complete",
}
PHASE_ICONS = {
    "discovery":  "🔍",
    "alignment":  "🤝",
    "design":     "📐",
    "refinement": "✏️",
    "complete":   "✅",
}


def _normalise_phase(raw: str) -> str:
    """Accept any casing from graph state and return a canonical lowercase key."""
    return raw.strip().lower() if isinstance(raw, str) else "discovery"


def _phase_index(phase: str) -> int:
    """Safe PHASE_ORDER lookup — returns -1 if the value is not recognised."""
    try:
        return PHASE_ORDER.index(_normalise_phase(phase))
    except ValueError:
        return -1


# ── Sessions persistence ──────────────────────────────────────────────────────
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
        "phase": st.session_state.phase,          # stored as lowercase
        "plan_draft": st.session_state.plan_draft,
        "approved": st.session_state.approved,
        "awaiting_resume": st.session_state.awaiting_resume,
        "updated_at": datetime.datetime.now().isoformat(),
    }
    _save_all_sessions(sessions)


def _switch_session(thread_id: str):
    sessions = _load_all_sessions()
    s = sessions.get(thread_id)
    if s:
        st.session_state.thread_id = s["thread_id"]
        st.session_state.chat_history = s["chat_history"]
        # Normalise phase coming from disk — old sessions may have Title-case values
        st.session_state.phase = _normalise_phase(s.get("phase", "discovery"))
        st.session_state.plan_draft = s.get("plan_draft")
        st.session_state.approved = s.get("approved", False)
        st.session_state.awaiting_resume = s.get("awaiting_resume", False)
    st.session_state.renaming_session = None
    st.rerun()


# ── Session state initialisation ──────────────────────────────────────────────
def _init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "phase" not in st.session_state:
        st.session_state.phase = "discovery"       # lowercase to match graph.py
    if "plan_draft" not in st.session_state:
        st.session_state.plan_draft = None
    if "awaiting_resume" not in st.session_state:
        st.session_state.awaiting_resume = False
    if "approved" not in st.session_state:
        st.session_state.approved = False
    if "renaming_session" not in st.session_state:
        st.session_state.renaming_session = None

_init_session()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _sync_state_from_graph():
    """Pull the latest phase / plan / approved flag out of the graph checkpoint."""
    try:
        snapshot = graph.get_state(_get_config())
        if snapshot and snapshot.values:
            raw_phase = snapshot.values.get("phase", st.session_state.phase)
            st.session_state.phase = _normalise_phase(raw_phase)

            plan = snapshot.values.get("plan_draft")
            if plan:
                st.session_state.plan_draft = plan

            if snapshot.values.get("approved"):
                st.session_state.approved = True
                st.session_state.phase = "complete"
    except Exception:
        pass


def _run_graph(user_message: str) -> str:
    config = _get_config()
    label = PHASE_LABELS.get(st.session_state.phase, st.session_state.phase.title())

    with st.spinner(f"Agent working — {label} phase..."):
        try:
            if st.session_state.awaiting_resume:
                snapshot = graph.get_state(config)
                if not snapshot or not snapshot.values:
                    # Checkpoint lost — restart cleanly
                    st.session_state.awaiting_resume = False
                    result = graph.invoke(
                        {"messages": [HumanMessage(content=user_message)]},
                        config=config,
                    )
                else:
                    result = graph.invoke(Command(resume=user_message), config=config)
            else:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_message)]},
                    config=config,
                )

            st.session_state.awaiting_resume = False
            _sync_state_from_graph()

            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
                    return msg.content
            return result.get("final_plan") or result.get("plan_draft") or "Done."

        except GraphInterrupt as exc:
            st.session_state.awaiting_resume = True
            _sync_state_from_graph()
            interrupted_value = exc.args[0] if exc.args else ""
            return str(interrupted_value)

        except Exception as exc:
            st.session_state.awaiting_resume = False
            return f"⚠️ Error: {exc}"


def _new_session():
    for key in ["thread_id", "chat_history", "phase", "plan_draft", "awaiting_resume", "approved"]:
        st.session_state.pop(key, None)
    st.session_state.renaming_session = None
    _init_session()
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Planning Agent")
    st.caption("A multi-phase planning assistant powered by LLMs and LangGraph.")

    st.divider()

    # Phase progress — compare using _phase_index so casing never matters
    st.subheader("Phase")
    current_phase = _normalise_phase(st.session_state.phase)
    current_idx = _phase_index(current_phase)

    for phase_key in PHASE_ORDER:
        icon  = PHASE_ICONS[phase_key]
        label = PHASE_LABELS[phase_key]
        idx   = _phase_index(phase_key)

        if phase_key == current_phase:
            st.markdown(f"**→ {icon} {label}**")
        elif idx < current_idx:
            st.markdown(f"~~{icon} {label}~~")
        else:
            st.markdown(f"{icon} {label}")

    st.divider()

    if st.button("➕ New Session", use_container_width=True):
        _new_session()

    st.divider()

    # Past sessions
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

    if st.session_state.approved:
        st.divider()
        st.success("Plan approved!")

# ── Main layout ───────────────────────────────────────────────────────────────
chat_col, plan_col = st.columns([6, 4])

# ── Left column: Chat ─────────────────────────────────────────────────────────
with chat_col:
    st.header("Chat")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.approved:
        st.info("The plan has been approved. Start a new session to plan something else.")
    else:
        placeholder = (
            "Reply to the agent..." if st.session_state.awaiting_resume
            else "Describe what you want to plan..."
        )
        user_input = st.chat_input(placeholder)

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            agent_response = _run_graph(user_input)

            st.session_state.chat_history.append({"role": "assistant", "content": agent_response})
            with st.chat_message("assistant"):
                st.markdown(agent_response)

            _save_current_session()

            if st.session_state.phase in ("design", "refinement", "complete"):
                _sync_state_from_graph()

            st.rerun()

# ── Right column: Plan preview ────────────────────────────────────────────────
with plan_col:
    st.header("Plan Preview")

    if st.session_state.plan_draft:
        st.markdown(st.session_state.plan_draft)

        st.divider()

        if st.session_state.phase in ("design", "refinement") and not st.session_state.approved:
            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                if st.button("✅ Approve Plan", use_container_width=True, type="primary"):
                    approval_response = _run_graph("Approved")
                    st.session_state.chat_history.append({"role": "user", "content": "Approved"})
                    st.session_state.chat_history.append({"role": "assistant", "content": approval_response})
                    _save_current_session()
                    _sync_state_from_graph()
                    st.rerun()

            with btn_col2:
                if st.button("✏️ Request Changes", use_container_width=True):
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": "What changes would you like to make to the plan?"}
                    )
                    st.rerun()
    else:
        st.info("The plan will appear here once the Design phase is complete.")