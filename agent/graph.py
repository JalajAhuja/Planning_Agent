from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt



from agent.formatter import save_plan
from agent.prompts import (
    alignment_prompt,
    design_prompt,
    discovery_prompt,
    refinement_prompt,
)
from agent.state import FeedbackClassification, PlannerState
from agent.tools import tools

load_dotenv()

# ── LLM instances ─────────────────────────────────────────────────────────────
_discovery_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)
_alignment_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
_design_llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
_refinement_llm = ChatOpenAI(model="gpt-4o", temperature=0.2).with_structured_output(
    FeedbackClassification
)


# ── Node: discovery ───────────────────────────────────────────────────────────
def discovery(state: PlannerState) -> dict:
    """
    Phase 1 — Silent ReAct research loop.

    Runs entirely in the background with no interrupts and no output shown to the
    user. When the LLM stops calling tools it stores the research summary and
    advances phase to 'alignment'.
    """
    messages = list(state["messages"])

    # Prepend the system prompt on the very first call only.
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=discovery_prompt)] + messages

    response: AIMessage = _discovery_llm.invoke(messages)
    updates: dict = {"messages": [response]}

    if not response.tool_calls:
        updates["research_summary"] = response.content
        updates["phase"] = "alignment"

    return updates


# ── Node: tool_executor ───────────────────────────────────────────────────────
tool_executor = ToolNode(tools, handle_tool_errors=True)


def _route_discovery(state: PlannerState) -> Literal["tool_executor", "alignment"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_executor"
    return "alignment"


# ── Node: alignment ───────────────────────────────────────────────────────────
def alignment(state: PlannerState) -> dict:
    """
    Phase 2 — Ask ONE round of clarifying questions (each with a skip option).

    If the LLM already has enough context it returns PROCEED_TO_DESIGN and we
    skip the interrupt entirely. Otherwise we pause once for the user's answers
    and immediately advance to design — no second round.
    """
    research = state.get("research_summary", "")

    messages = [
        SystemMessage(content=alignment_prompt),
        HumanMessage(content=f"Research summary:\n\n{research}"),
    ]

    try:
        response: AIMessage = _alignment_llm.invoke(messages)
        questions_text = (response.content or "").strip()
    except Exception:
        return {
            "clarifying_questions": [],
            "user_answers": [],
            "phase": "design",
        }

    # LLM decided it has enough info — skip the interrupt
    if "PROCEED_TO_DESIGN" in questions_text.upper():
        return {
            "messages": [response],
            "clarifying_questions": [],
            "user_answers": [],
            "phase": "design",
        }

    # Single interrupt: ask questions, collect answers, advance to design
    user_answer: str = interrupt(questions_text)

    return {
        "messages": [response],
        "clarifying_questions": [questions_text],
        "user_answers": [user_answer],
        "phase": "design",
    }


# ── Node: design ──────────────────────────────────────────────────────────────
def design(state: PlannerState) -> dict:
    """
    Phase 3 — Generate the plan and commit it to state.

    No interrupt here. plan_draft is written to state before present_plan runs,
    so the Plan Preview panel in the UI can read it from the checkpoint snapshot
    while the user is being asked for feedback.
    """
    messages = [
        SystemMessage(content=design_prompt),
        HumanMessage(content=f"Research summary:\n\n{state.get('research_summary', '')}"),
        HumanMessage(
            content=f"Clarifying answers from user:\n\n{state.get('user_answers', [])}"
        ),
    ]

    try:
        response: AIMessage = _design_llm.invoke(messages)
        plan_draft = response.content
    except Exception as exc:
        plan_draft = _fallback_plan(state, str(exc))

    save_plan(plan_draft)

    return {
        "plan_draft": plan_draft,
        "phase": "refinement",
        "refinement_action": None,   # reset so present_plan → refinement is a clean entry
        "display_content": plan_draft,  # picked up by present_plan
    }


def _fallback_plan(state: PlannerState, error_hint: str) -> str:
    research = state.get("research_summary", "No research available.")
    answers = state.get("user_answers", [])
    answers_str = "\n".join(f"- {a}" for a in answers) if answers else "None provided."
    return (
        "## Plan (fallback — LLM unavailable)\n\n"
        f"**Error:** {error_hint[:200]}\n\n"
        "### Research Summary\n"
        f"{research}\n\n"
        "### User Clarifications\n"
        f"{answers_str}\n\n"
        "### Suggested Next Steps\n"
        "1. Review the research summary above.\n"
        "2. Incorporate the clarifications from the alignment phase.\n"
        "3. Proceed with implementation iteratively.\n"
    )


# ── Node: present_plan ────────────────────────────────────────────────────────
def present_plan(state: PlannerState) -> dict:
    """
    Single-responsibility presentation node — the ONLY place that interrupts.

    Shows display_content (or plan_draft as fallback) to the user, collects
    their response, and stores it as a HumanMessage for refinement to classify.
    display_content is cleared after use so it isn't re-shown on the next pass.
    """
    content = state.get("display_content") or state.get("plan_draft", "No plan available yet.")
    user_input: str = interrupt(content)
    return {
        "messages": [HumanMessage(content=user_input)],
        "display_content": None,
    }


def _route_after_present_plan(
    state: PlannerState,
) -> Literal["refinement", "discovery"]:
    """
    Route to discovery when an alternative restart was confirmed by the user,
    otherwise always route to refinement for feedback classification.
    """
    if state.get("phase") == "discovery":
        return "discovery"
    return "refinement"


# ── Node: refinement ─────────────────────────────────────────────────────────
def refinement(state: PlannerState) -> dict:
    """
    Phase 4 — Classify user feedback with structured output and update state.

    No interrupts here; all user interaction is handled by present_plan.
    Writes refinement_action and (where needed) display_content so that
    _route_refinement and present_plan know what to show next.
    """
    plan_draft = state.get("plan_draft", "")
    user_feedback = state["messages"][-1].content

    messages = [
        SystemMessage(content=refinement_prompt),
        HumanMessage(content=f"Current plan:\n\n{plan_draft}"),
        HumanMessage(content=f"User feedback:\n\n{user_feedback}"),
    ]

    try:
        result: FeedbackClassification = _refinement_llm.invoke(messages)
        feedback_type = result.feedback_type
    except Exception:
        return {
            "refinement_action": "question",
            "display_content": (
                "I had trouble understanding your feedback. Could you rephrase it?\n\n"
                "(e.g. 'Approved', 'Change X to Y', or 'Try a completely different approach')"
            ),
        }

    if feedback_type == "approval":
        return {
            "messages": [AIMessage(content=(
                "Plan approved. You can now hand this off to an implementation agent "
                "or proceed step by step. The plan has been saved."
            ))],
            "approved": True,
            "final_plan": plan_draft,
            "phase": "complete",
            "refinement_action": "approve",
        }

    if feedback_type == "revision":
        updated_plan = result.updated_plan or plan_draft
        save_plan(updated_plan)
        return {
            "plan_draft": updated_plan,
            "phase": "refinement",
            "refinement_action": "revise",
            "display_content": updated_plan,
        }

    if feedback_type == "question":
        answer = result.answer or "I don't have more details on that."
        return {
            "phase": "refinement",
            "refinement_action": "question",
            "display_content": answer,
        }

    if feedback_type == "alternative":
        return {
            "phase": "discovery",
            "research_summary": None,
            "plan_draft": None,
            "clarifying_questions": [],
            "user_answers": [],
            "refinement_action": "alternative",
            "display_content": (
                "Got it — I'll restart research with a new direction.\n\n"
                "Please confirm or describe what you'd like to focus on instead."
            ),
        }

    # Fallback
    return {
        "refinement_action": "question",
        "display_content": "I didn't understand your feedback. Please rephrase.",
    }


def _route_refinement(
    state: PlannerState,
) -> Literal["present_plan", "__end__"]:
    if state.get("refinement_action") == "approve":
        return END
    return "present_plan"


# ── Graph assembly ────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(PlannerState)

    builder.add_node("discovery", discovery)
    builder.add_node("tool_executor", tool_executor)
    builder.add_node("alignment", alignment)
    builder.add_node("design", design)
    builder.add_node("present_plan", present_plan)
    builder.add_node("refinement", refinement)

    # Entry
    builder.add_edge(START, "discovery")

    # Discovery ReAct loop: tool calls feed back; final text → alignment
    builder.add_conditional_edges(
        "discovery",
        _route_discovery,
        {"tool_executor": "tool_executor", "alignment": "alignment"},
    )
    builder.add_edge("tool_executor", "discovery")

    # Alignment always produces one round then flows straight to design (no loop)
    builder.add_edge("alignment", "design")

    # Design commits plan_draft to state, then present_plan shows it to the user
    builder.add_edge("design", "present_plan")

    # After user input: route to refinement or restart discovery (alternative path)
    builder.add_conditional_edges(
        "present_plan",
        _route_after_present_plan,
        {"refinement": "refinement", "discovery": "discovery"},
    )

    # Refinement: approve → END, everything else → show updated content via present_plan
    builder.add_conditional_edges(
        "refinement",
        _route_refinement,
        {"present_plan": "present_plan", END: END},
    )

    # Persistence
    db_path = Path(__file__).parent.parent / "checkpoints.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()