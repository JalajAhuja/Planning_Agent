import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from agent.graph import graph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Planning Agent",
    page_icon="🗺️",
    layout="wide",
)

# ── Session state initialisation ──────────────────────────────────────────────
def _init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        # list of {"role": "user"|"assistant", "content": str}
        st.session_state.chat_history = []
    if "phase" not in st.session_state:
        st.session_state.phase = "Discovery"
    if "plan_draft" not in st.session_state:
        st.session_state.plan_draft = None
    if "awaiting_resume" not in st.session_state:
        # True when graph has been interrupted and is waiting for user input
        st.session_state.awaiting_resume = False
    if "approved" not in st.session_state:
        st.session_state.approved = False

_init_session()

# ── Helpers ───────────────────────────────────────────────────────────────────
PHASE_ORDER = ["Discovery", "Alignment", "Design", "Refinement", "Complete"]
PHASE_ICONS = {
    "Discovery": "🔍",
    "Alignment": "🤝",
    "Design": "📐",
    "Refinement": "✏️",
    "Complete": "✅",
}


def _get_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _sync_state_from_graph():
    """Pull phase and plan_draft from the latest graph snapshot into session state."""
    try:
        snapshot = graph.get_state(_get_config())
        if snapshot and snapshot.values:
            st.session_state.phase = snapshot.values.get("phase", st.session_state.phase)
            plan = snapshot.values.get("plan_draft")
            if plan:
                st.session_state.plan_draft = plan
            if snapshot.values.get("approved"):
                st.session_state.approved = True
                st.session_state.phase = "Complete"
    except Exception:
        pass


def _run_graph(user_message: str):
    """Send a message into the graph (first turn) or resume after an interrupt."""
    config = _get_config()

    with st.spinner(f"Agent working — {st.session_state.phase} phase..."):
        try:
            if st.session_state.awaiting_resume:
                # Resume the interrupted graph with the user's answer
                result = graph.invoke(
                    Command(resume=user_message),
                    config=config,
                )
            else:
                # First invocation — start the graph
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_message)]},
                    config=config,
                )

            # Graph ran to END without interrupting
            st.session_state.awaiting_resume = False
            _sync_state_from_graph()

            # Extract the last assistant message to display
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
                    return msg.content
            return result.get("final_plan") or result.get("plan_draft") or "Done."

        except GraphInterrupt as exc:
            # Graph paused at an interrupt() call; exc.args[0] is the surfaced value
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
    _init_session()
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Planning Agent")
    st.caption("A multi-phase planning assistant powered by LLMs and LangGraph.")

    st.divider()

    # Phase progress
    st.subheader("Phase")
    current_phase = st.session_state.phase
    for phase in PHASE_ORDER:
        icon = PHASE_ICONS[phase]
        if phase == current_phase:
            st.markdown(f"**→ {icon} {phase}**")
        elif PHASE_ORDER.index(phase) < PHASE_ORDER.index(current_phase):
            st.markdown(f"~~{icon} {phase}~~")
        else:
            st.markdown(f"{icon} {phase}")

    st.divider()

    # Thread info
    st.caption(f"Session: `{st.session_state.thread_id[:8]}...`")
    if st.button("🔄 New Session", use_container_width=True):
        _new_session()

    if st.session_state.approved:
        st.success("Plan approved!")

# ── Main layout ───────────────────────────────────────────────────────────────
chat_col, plan_col = st.columns([6, 4])

# ── Left column: Chat ─────────────────────────────────────────────────────────
with chat_col:
    st.header("Chat")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input area
    if st.session_state.approved:
        st.info("The plan has been approved. Start a new session to plan something else.")
    else:
        placeholder = (
            "Reply to the agent..." if st.session_state.awaiting_resume
            else "Describe what you want to plan..."
        )
        user_input = st.chat_input(placeholder)

        if user_input:
            # Show user message immediately
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Run graph and get response
            agent_response = _run_graph(user_input)

            # Show agent response
            st.session_state.chat_history.append({"role": "assistant", "content": agent_response})
            with st.chat_message("assistant"):
                st.markdown(agent_response)

            # Update plan preview if the response looks like a plan
            if st.session_state.phase in ("Design", "Refinement", "Complete"):
                _sync_state_from_graph()

            st.rerun()

# ── Right column: Plan preview ────────────────────────────────────────────────
with plan_col:
    st.header("Plan Preview")

    if st.session_state.plan_draft:
        st.markdown(st.session_state.plan_draft)

        st.divider()

        # Action buttons — only visible during Refinement
        if st.session_state.phase in ("Design", "Refinement") and not st.session_state.approved:
            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                if st.button("✅ Approve Plan", use_container_width=True, type="primary"):
                    approval_response = _run_graph("Approved")
                    st.session_state.chat_history.append(
                        {"role": "user", "content": "Approved"}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": approval_response}
                    )
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
        st.info("The plan will appear here once the Design phase is complete.")
