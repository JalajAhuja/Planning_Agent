"""Tests for agent/routers.py"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.routers import (
    route_after_alignment_decide,
    route_after_discovery,
    route_after_refinement,
)


# ── route_after_discovery ──────────────────────────────────────────────────────

def _make_ai_with_tools():
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": "tavily_search", "args": {"query": "test"}, "id": "1"}]
    return msg


def test_route_after_discovery_tool_calls():
    """Routes to tool_executor when last AIMessage has tool calls."""
    state = {"messages": [HumanMessage(content="plan a trip"), _make_ai_with_tools()]}
    assert route_after_discovery(state) == "tool_executor"


def test_route_after_discovery_no_tool_calls():
    """Routes to research_summary when last AIMessage has no tool calls."""
    state = {"messages": [HumanMessage(content="plan a trip"), AIMessage(content="Research complete.")]}
    assert route_after_discovery(state) == "research_summary"


def test_route_after_discovery_plain_ai_message():
    """A plain AIMessage with empty tool_calls routes to research_summary."""
    msg = AIMessage(content="Research complete.")
    msg.tool_calls = []
    state = {"messages": [msg]}
    assert route_after_discovery(state) == "research_summary"


# ── route_after_alignment_decide ───────────────────────────────────────────────

def test_route_after_alignment_decide_needs_clarification():
    """Routes to alignment_collect when need_clarification=True and questions exist."""
    state = {
        "need_clarification": True,
        "clarifying_questions": "Q1: What is your timeline?",
    }
    assert route_after_alignment_decide(state) == "alignment_collect"


def test_route_after_alignment_decide_no_clarification():
    """Routes to design when need_clarification=False."""
    state = {
        "need_clarification": False,
        "clarifying_questions": "",
    }
    assert route_after_alignment_decide(state) == "design"


def test_route_after_alignment_decide_clarification_true_empty_questions():
    """Routes to design when need_clarification=True but questions is empty string."""
    state = {
        "need_clarification": True,
        "clarifying_questions": "",
    }
    assert route_after_alignment_decide(state) == "design"


def test_route_after_alignment_decide_missing_keys():
    """Routes to design when state keys are absent (falsy defaults)."""
    assert route_after_alignment_decide({}) == "design"


# ── route_after_refinement ─────────────────────────────────────────────────────

def test_route_after_refinement_approved():
    """Routes to __end__ when approved=True."""
    state = {"approved": True, "plan_draft": "some plan"}
    assert route_after_refinement(state) == "__end__"


def test_route_after_refinement_no_plan():
    """Routes to discovery when plan_draft is None (alternative feedback)."""
    state = {"approved": False, "plan_draft": None}
    assert route_after_refinement(state) == "discovery"


def test_route_after_refinement_no_plan_missing_key():
    """Routes to discovery when plan_draft key is absent."""
    state = {"approved": False}
    assert route_after_refinement(state) == "discovery"


def test_route_after_refinement_has_plan():
    """Routes to present_plan when plan_draft exists and not approved."""
    state = {"approved": False, "plan_draft": "## Plan: Test\n..."}
    assert route_after_refinement(state) == "present_plan"


def test_route_after_refinement_empty_state():
    """Routes to discovery when state is completely empty."""
    assert route_after_refinement({}) == "discovery"
