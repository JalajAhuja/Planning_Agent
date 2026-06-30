from typing import Literal

from langchain_core.messages import AIMessage

from agent.state import PlannerState


def route_after_discovery(
    state: PlannerState,
) -> Literal["tool_executor", "research_summary"]:
    """If the last Discovery message contains tool calls, run them.
    Otherwise move on to ResearchSummary."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_executor"
    return "research_summary"


def route_after_refinement(
    state: PlannerState,
) -> Literal["__end__", "present_plan", "discovery"]:
    """Approval → END.  Alternative (no plan_draft) → Discovery.  Otherwise → PresentPlan."""
    if state.get("approved"):
        return "__end__"
    if not state.get("plan_draft"):
        return "discovery"
    return "present_plan"
