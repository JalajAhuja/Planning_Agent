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


# ── Node: alignment_decide ───────────────────────────────────────────────────

def alignment_decide(state: PlannerState) -> dict:
    """Phase 2a — LLM decides whether clarification is needed.

    Commits the decision (need_clarification, clarifying_questions) to the
    checkpoint so that alignment_collect can read it on the next node run.
    This node never calls interrupt() — that keeps it safe from the LangGraph
    re-run-on-resume behaviour.
    """
    # Include the original user request so the LLM knows the topic.
    msgs = state.get("messages", [])
    original_request = next(
        (m.content for m in msgs if isinstance(m, HumanMessage)), ""
    )

    decision = alignment_llm.invoke(
        [
            SystemMessage(content=alignment_prompt),
            HumanMessage(
                content=(
                    f"Original Request:\n{original_request}\n\n"
                    f"Research Summary:\n{state.get('research_summary', '')}"
                )
            ),
        ]
    )

    return {
        "need_clarification": decision.need_clarification,
        "clarifying_questions": decision.clarifying_questions,
    }


# ── Node: alignment_collect ───────────────────────────────────────────────────

def alignment_collect(state: PlannerState) -> dict:
    """Phase 2b — Interrupt to collect user answers.

    This node contains the ONLY interrupt() call for alignment.  It is
    intentionally free of any LLM calls so that re-running from the top on
    resume is completely safe: the questions are already committed to state by
    alignment_decide, and interrupt() simply returns the resume value.
    """
    questions = state.get("clarifying_questions", "")
    answers = interrupt(questions)

    if isinstance(answers, str):
        answers = [answers]

    return {"user_answers": answers}


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
