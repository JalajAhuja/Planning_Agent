"""
Tests for the alignment interrupt/resume flow — the key bug that was fixed.

These tests verify that:
1. alignment_decide commits its decision to state (no interrupt).
2. alignment_collect is the ONLY node that calls interrupt().
3. On resume, alignment_collect re-reads committed questions and returns the answer.
4. route_after_alignment_decide gates whether alignment_collect is called at all.

The tests simulate the LangGraph checkpoint pattern by manually threading state
through the two nodes, matching what the graph would do.
"""
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from agent.routers import route_after_alignment_decide
from agent.state import AlignmentDecision


def _make_state(*, need_clarification: bool, questions: str = "", user_msg: str = "Build a React app"):
    return {
        "messages": [HumanMessage(content=user_msg)],
        "research_summary": "React is a JavaScript library for building UIs.",
        "need_clarification": need_clarification,
        "clarifying_questions": questions,
    }


# ── Scenario A: No clarification needed ───────────────────────────────────────

class TestAlignmentFlowNoClarification:
    def test_decide_returns_no_clarification(self):
        """alignment_decide sets need_clarification=False when LLM says no questions needed."""
        decision = AlignmentDecision(need_clarification=False, clarifying_questions="")
        with patch("agent.nodes.alignment_llm") as mock_llm:
            mock_llm.invoke.return_value = decision
            from agent.nodes import alignment_decide
            result = alignment_decide(_make_state(need_clarification=False))

        assert result["need_clarification"] is False
        assert result["clarifying_questions"] == ""

    def test_router_skips_collect_when_no_clarification(self):
        """route_after_alignment_decide routes to design, skipping alignment_collect."""
        state = _make_state(need_clarification=False, questions="")
        state.update({"need_clarification": False, "clarifying_questions": ""})
        assert route_after_alignment_decide(state) == "design"

    def test_collect_node_never_reached(self):
        """Simulate the full flow: decide → route → no collect called."""
        decision = AlignmentDecision(need_clarification=False)
        state = _make_state(need_clarification=False)

        with patch("agent.nodes.alignment_llm") as mock_llm:
            mock_llm.invoke.return_value = decision
            from agent.nodes import alignment_decide
            updates = alignment_decide(state)

        # Merge updates into state (as LangGraph would)
        state.update(updates)
        route = route_after_alignment_decide(state)
        assert route == "design"
        # alignment_collect is never reached — no interrupt fired


# ── Scenario B: Clarification needed — first run (interrupt fires) ─────────────

class TestAlignmentFlowFirstRun:
    def test_decide_sets_questions_in_state(self):
        """alignment_decide commits questions to state on first run."""
        questions = "**Q1: Auth method?**\n1. JWT\n2. Session\n3. Skip"
        decision = AlignmentDecision(need_clarification=True, clarifying_questions=questions)

        with patch("agent.nodes.alignment_llm") as mock_llm:
            mock_llm.invoke.return_value = decision
            from agent.nodes import alignment_decide
            result = alignment_decide(_make_state(need_clarification=True))

        assert result["need_clarification"] is True
        assert result["clarifying_questions"] == questions

    def test_router_sends_to_collect_when_clarification_needed(self):
        """route_after_alignment_decide routes to alignment_collect."""
        state = _make_state(
            need_clarification=True,
            questions="**Q1: Auth method?**",
        )
        state.update({"need_clarification": True})
        assert route_after_alignment_decide(state) == "alignment_collect"

    def test_collect_calls_interrupt_with_committed_questions(self):
        """alignment_collect calls interrupt() using the questions already in state."""
        questions = "**Q1: Auth method?**\n1. JWT\n2. Session\n3. Skip"
        state = _make_state(need_clarification=True, questions=questions)

        with patch("agent.nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "1. JWT"
            from agent.nodes import alignment_collect
            result = alignment_collect(state)

        mock_interrupt.assert_called_once_with(questions)
        assert result["user_answers"] == ["1. JWT"]


# ── Scenario C: Resume after interrupt ────────────────────────────────────────

class TestAlignmentFlowResume:
    """
    Simulates LangGraph's resume behaviour:
    - Checkpoint has state from AFTER alignment_decide ran (questions committed).
    - alignment_collect re-runs from the top.
    - interrupt() returns the resume value immediately (no new pause).
    """

    def test_collect_is_idempotent_on_resume(self):
        """
        On resume, alignment_collect re-runs with the same committed questions.
        interrupt() returns the user's answer and the node completes normally.
        The LLM is never called — so there is no risk of a different decision.
        """
        questions = "**Q1: Auth method?**\n1. JWT\n2. Session"
        # Simulate checkpoint state (alignment_decide already ran and committed)
        resumed_state = {
            "messages": [HumanMessage(content="Build a React app")],
            "research_summary": "React is a JS library.",
            "need_clarification": True,
            "clarifying_questions": questions,  # already committed
        }

        with patch("agent.nodes.interrupt", return_value="1. JWT") as mock_interrupt:
            with patch("agent.nodes.alignment_llm") as mock_llm:
                from agent.nodes import alignment_collect
                result = alignment_collect(resumed_state)

        # interrupt was called with the committed questions
        mock_interrupt.assert_called_once_with(questions)
        # LLM was NOT called — no risk of different decision
        mock_llm.invoke.assert_not_called()
        # User answer was collected correctly
        assert result["user_answers"] == ["1. JWT"]

    def test_collect_never_calls_llm(self):
        """alignment_collect must not call any LLM, making it safe to re-run."""
        state = _make_state(need_clarification=True, questions="Q1?")
        with patch("agent.nodes.interrupt", return_value="answer"):
            with patch("agent.nodes.alignment_llm") as mock_llm:
                from agent.nodes import alignment_collect
                alignment_collect(state)
                mock_llm.invoke.assert_not_called()

    def test_decide_only_runs_once_not_on_resume(self):
        """
        This test documents the bug that was fixed:
        The OLD alignment node called alignment_llm BEFORE interrupt(), meaning
        on resume the LLM could return need_clarification=False and skip interrupt().

        With the new split design, alignment_decide runs once and commits its
        decision. alignment_collect only runs interrupt() — no LLM call.
        The LLM decision is stable across the interrupt boundary.
        """
        questions = "**Q1: Auth method?**"
        state = {
            "messages": [HumanMessage(content="Build a React app")],
            "research_summary": "React summary.",
            "need_clarification": True,
            "clarifying_questions": questions,
        }

        llm_call_count = 0

        def counting_interrupt(value):
            return "user answer"

        with patch("agent.nodes.interrupt", side_effect=counting_interrupt):
            with patch("agent.nodes.alignment_llm") as mock_llm:
                from agent.nodes import alignment_collect
                result = alignment_collect(state)
                # LLM should NOT be called inside alignment_collect
                assert mock_llm.invoke.call_count == 0

        assert result["user_answers"] == ["user answer"]


# ── Scenario D: Edge cases ─────────────────────────────────────────────────────

class TestAlignmentEdgeCases:
    def test_collect_wraps_non_list_answer(self):
        """interrupt() returning a plain string is wrapped into a list."""
        state = _make_state(need_clarification=True, questions="Q1?")
        with patch("agent.nodes.interrupt", return_value="plain string answer"):
            from agent.nodes import alignment_collect
            result = alignment_collect(state)
        assert result["user_answers"] == ["plain string answer"]

    def test_collect_preserves_list_answer(self):
        """interrupt() returning a list is kept as-is."""
        state = _make_state(need_clarification=True, questions="Q1? Q2?")
        with patch("agent.nodes.interrupt", return_value=["ans 1", "ans 2"]):
            from agent.nodes import alignment_collect
            result = alignment_collect(state)
        assert result["user_answers"] == ["ans 1", "ans 2"]

    def test_decide_handles_empty_messages(self):
        """alignment_decide handles state with empty messages list (no crash)."""
        decision = AlignmentDecision(need_clarification=False)
        state = {"messages": [], "research_summary": "Summary here."}
        with patch("agent.nodes.alignment_llm") as mock_llm:
            mock_llm.invoke.return_value = decision
            from agent.nodes import alignment_decide
            result = alignment_decide(state)
        assert "need_clarification" in result

    def test_router_falsy_questions_routes_to_design(self):
        """Even if need_clarification=True, empty questions routes to design (safe fallback)."""
        state = {"need_clarification": True, "clarifying_questions": ""}
        assert route_after_alignment_decide(state) == "design"
