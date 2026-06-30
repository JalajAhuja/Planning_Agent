from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from agent.formatter import save_plan
from agent.llms import (
    alignment_llm,
    design_llm,
    discovery_summary_llm,
    discovery_tool_llm,
    refinement_llm,
)
from agent.prompts import (
    alignment_prompt,
    design_prompt,
    discovery_prompt,
    refinement_prompt,
    research_summary_prompt,
)
from agent.state import PlannerState


# ── Node: discovery ───────────────────────────────────────────────────────────

def discovery(state: PlannerState) -> dict:
    """Phase 1 — Research only.

    Calls tools to gather information about the user's topic.
    Never summarizes. Never talks to the UI. Returns AIMessage only.
    """
    response: AIMessage = discovery_tool_llm.invoke(
        [SystemMessage(content=discovery_prompt), *state["messages"]]
    )
    return {"messages": [response]}


# ── Node: research_summary ────────────────────────────────────────────────────

def research_summary(state: PlannerState) -> dict:
    """Reads all Discovery messages and produces a structured RouterDecision.

    Sets research_summary, need_clarification, and clarifying_questions.
    Never emits a user-visible message.
    """
    summary = discovery_summary_llm.invoke(
        [SystemMessage(content=research_summary_prompt), *state["messages"]]
    )
    return {
        "research_summary": summary.research_summary,
        "need_clarification": summary.need_clarification,
        "clarifying_questions": summary.clarifying_questions,
    }


# ── Node: alignment ───────────────────────────────────────────────────────────

def alignment(state: PlannerState) -> dict:
    """Phase 2 — Ask ONE round of clarification if needed.

    If no clarification needed, continues directly to Design.
    Uses interrupt() to pause for user answers.
    """
    decision = alignment_llm.invoke(
        [
            SystemMessage(content=alignment_prompt),
            HumanMessage(
                content=f"Research Summary:\n{state.get('research_summary', '')}"
            ),
        ]
    )

    if not decision.need_clarification:
        return {
            "need_clarification": False,
            "clarifying_questions": "",
            "user_answers": [],
        }

    answers = interrupt(decision.clarifying_questions)

    if isinstance(answers, str):
        answers = [answers]

    return {
        "need_clarification": True,
        "clarifying_questions": decision.clarifying_questions,
        "user_answers": answers,
    }


# ── Node: design ──────────────────────────────────────────────────────────────

def design(state: PlannerState) -> dict:
    """Phase 3 — Produce a complete, structured implementation plan."""
    response = design_llm.invoke(
        [
            SystemMessage(content=design_prompt),
            HumanMessage(
                content=(
                    f"Research Summary:\n{state.get('research_summary', '')}\n\n"
                    f"Clarifying Questions:\n{state.get('clarifying_questions', '')}\n\n"
                    f"User Answers:\n{state.get('user_answers', [])}"
                )
            ),
        ]
    )
    return {"plan_draft": response.content}


# Shared prompt shown in chat when the plan is ready.
# Must match the value passed to interrupt() in refinement so the dedup
# logic in app.py never shows it twice.
_PLAN_READY = (
    "Your plan is ready! Review it in the **Plan Preview** panel on the right.\n\n"
    "Reply here with feedback, request revisions, or type **approve** to finalize."
)


# ── Node: present_plan ────────────────────────────────────────────────────────

def present_plan(state: PlannerState) -> dict:
    """Emits a short user-facing message — the ONLY user-visible node.

    The full plan lives in plan_draft / the Plan Preview panel, NOT in chat.
    """
    return {"messages": [AIMessage(content=_PLAN_READY)]}


# ── Node: refinement ──────────────────────────────────────────────────────────

def refinement(state: PlannerState) -> dict:
    """Phase 4 — Collect feedback, classify it, act on it.

    Interrupts to present the plan and wait for user feedback.
    Handles: approval, revision, question, alternative.
    """
    feedback = interrupt(_PLAN_READY)

    decision = refinement_llm.invoke(
        [
            SystemMessage(content=refinement_prompt),
            HumanMessage(
                content=(
                    f"Current Plan:\n{state['plan_draft']}\n\n"
                    f"User Feedback:\n{feedback}"
                )
            ),
        ]
    )

    # ── Approval ──────────────────────────────────────────────────────────────
    if decision.feedback_type == "approval":
        path = save_plan(state["plan_draft"])
        return {
            "approved": True,
            "final_plan": state["plan_draft"],
            "plan_file_path": path,
            "messages": [
                AIMessage(content=f"Plan approved and saved to `{path}`.")
            ],
        }

    # ── Question ──────────────────────────────────────────────────────────────
    if decision.feedback_type == "question":
        return {
            "approved": False,
            "messages": [AIMessage(content=decision.answer)],
        }

    # ── Revision ──────────────────────────────────────────────────────────────
    if decision.feedback_type == "revision":
        return {
            "plan_draft": decision.updated_plan,
            "approved": False,
        }

    # ── Alternative ───────────────────────────────────────────────────────────
    return {
        "research_summary": "",
        "clarifying_questions": "",
        "user_answers": [],
        "plan_draft": None,
        "approved": False,
    }
