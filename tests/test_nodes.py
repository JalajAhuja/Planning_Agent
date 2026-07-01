"""Tests for agent/nodes.py — unit tests with mocked LLMs and interrupt."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import AlignmentDecision, FeedbackClassification, RouterDecision


# ── Helpers ────────────────────────────────────────────────────────────────────

def _base_state(**extra):
    """Minimal PlannerState-like dict for tests."""
    return {
        "messages": [HumanMessage(content="Build a React app with auth")],
        "research_summary": "React is a JS library. Auth can use JWT or sessions.",
        **extra,
    }


# ── research_summary node ──────────────────────────────────────────────────────

class TestResearchSummary:
    def test_sets_all_required_fields(self):
        """research_summary node returns the three required state fields."""
        mock_summary = RouterDecision(
            research_summary="React best practices: hooks, context.",
            need_clarification=True,
            clarifying_questions="Q1: What auth method?",
        )
        with patch("agent.nodes.discovery_summary_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_summary
            from agent.nodes import research_summary
            result = research_summary(_base_state())

        assert result["research_summary"] == "React best practices: hooks, context."
        assert result["need_clarification"] is True
        assert result["clarifying_questions"] == "Q1: What auth method?"

    def test_no_clarification_returns_false(self):
        """When LLM says no clarification needed, need_clarification is False."""
        mock_summary = RouterDecision(
            research_summary="Enough info to proceed.",
            need_clarification=False,
            clarifying_questions="",
        )
        with patch("agent.nodes.discovery_summary_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_summary
            from agent.nodes import research_summary
            result = research_summary(_base_state())

        assert result["need_clarification"] is False
        assert result["clarifying_questions"] == ""

    def test_passes_messages_to_llm(self):
        """The full message list (including tool outputs) is passed to the LLM."""
        mock_summary = RouterDecision(
            research_summary="Summary.", need_clarification=False, clarifying_questions=""
        )
        state = _base_state()
        state["messages"].append(AIMessage(content="Research complete."))

        with patch("agent.nodes.discovery_summary_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_summary
            from agent.nodes import research_summary
            research_summary(state)

        call_args = mock_llm.invoke.call_args[0][0]
        # First arg is SystemMessage, rest are state messages
        assert isinstance(call_args[0], SystemMessage)
        # State messages should be in the call
        assert any(isinstance(m, HumanMessage) for m in call_args)


# ── alignment_decide node ──────────────────────────────────────────────────────

class TestAlignmentDecide:
    def test_no_clarification_returns_false(self):
        """When LLM says no clarification needed, returns need_clarification=False."""
        decision = AlignmentDecision(need_clarification=False, clarifying_questions="")
        with patch("agent.nodes.alignment_llm") as mock_llm:
            mock_llm.invoke.return_value = decision
            from agent.nodes import alignment_decide
            result = alignment_decide(_base_state())

        assert result["need_clarification"] is False
        assert result["clarifying_questions"] == ""

    def test_needs_clarification_returns_questions(self):
        """When LLM needs clarification, returns the questions in state."""
        questions = "**Q1: Auth method?**\n1. JWT\n2. Session\n3. Skip"
        decision = AlignmentDecision(need_clarification=True, clarifying_questions=questions)
        with patch("agent.nodes.alignment_llm") as mock_llm:
            mock_llm.invoke.return_value = decision
            from agent.nodes import alignment_decide
            result = alignment_decide(_base_state())

        assert result["need_clarification"] is True
        assert result["clarifying_questions"] == questions

    def test_includes_original_request_in_llm_call(self):
        """The first HumanMessage (original request) is forwarded to the LLM."""
        decision = AlignmentDecision(need_clarification=False)
        state = _base_state()

        with patch("agent.nodes.alignment_llm") as mock_llm:
            mock_llm.invoke.return_value = decision
            from agent.nodes import alignment_decide
            alignment_decide(state)

        call_msgs = mock_llm.invoke.call_args[0][0]
        human_msg = next(m for m in call_msgs if isinstance(m, HumanMessage))
        assert "Build a React app with auth" in human_msg.content

    def test_includes_research_summary_in_llm_call(self):
        """The research_summary field is forwarded to the LLM."""
        decision = AlignmentDecision(need_clarification=False)
        state = _base_state()

        with patch("agent.nodes.alignment_llm") as mock_llm:
            mock_llm.invoke.return_value = decision
            from agent.nodes import alignment_decide
            alignment_decide(state)

        call_msgs = mock_llm.invoke.call_args[0][0]
        human_msg = next(m for m in call_msgs if isinstance(m, HumanMessage))
        assert "React is a JS library" in human_msg.content

    def test_no_interrupt_called(self):
        """alignment_decide must never call interrupt()."""
        decision = AlignmentDecision(need_clarification=True, clarifying_questions="Q1?")
        with patch("agent.nodes.alignment_llm") as mock_llm:
            mock_llm.invoke.return_value = decision
            with patch("agent.nodes.interrupt") as mock_interrupt:
                from agent.nodes import alignment_decide
                alignment_decide(_base_state())
                mock_interrupt.assert_not_called()

    def test_missing_original_request_is_handled(self):
        """If messages list is empty, no crash — original_request is empty string."""
        decision = AlignmentDecision(need_clarification=False)
        state = {"messages": [], "research_summary": "Some summary."}

        with patch("agent.nodes.alignment_llm") as mock_llm:
            mock_llm.invoke.return_value = decision
            from agent.nodes import alignment_decide
            result = alignment_decide(state)  # must not raise

        assert "need_clarification" in result


# ── alignment_collect node ─────────────────────────────────────────────────────

class TestAlignmentCollect:
    def test_calls_interrupt_with_questions(self):
        """alignment_collect passes questions from state to interrupt()."""
        questions = "**Q1: Auth method?**\n1. JWT\n2. Session"
        state = _base_state(clarifying_questions=questions)

        with patch("agent.nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "1. JWT"
            from agent.nodes import alignment_collect
            result = alignment_collect(state)

        mock_interrupt.assert_called_once_with(questions)
        assert result["user_answers"] == ["1. JWT"]

    def test_string_answer_wrapped_in_list(self):
        """A plain string from interrupt() is wrapped into a list."""
        state = _base_state(clarifying_questions="Q1: Timeline?")
        with patch("agent.nodes.interrupt", return_value="3 months"):
            from agent.nodes import alignment_collect
            result = alignment_collect(state)

        assert result["user_answers"] == ["3 months"]

    def test_list_answer_passed_through(self):
        """A list from interrupt() is kept as-is."""
        state = _base_state(clarifying_questions="Q1?")
        with patch("agent.nodes.interrupt", return_value=["answer 1", "answer 2"]):
            from agent.nodes import alignment_collect
            result = alignment_collect(state)

        assert result["user_answers"] == ["answer 1", "answer 2"]

    def test_resume_idempotent_same_questions(self):
        """On resume, alignment_collect reads the same committed questions and returns the new answer."""
        questions = "**Q1: Framework?**"
        state = _base_state(clarifying_questions=questions)

        with patch("agent.nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "React"
            from agent.nodes import alignment_collect
            result = alignment_collect(state)

        # On re-run, same questions → interrupt returns the resume value
        assert mock_interrupt.call_args[0][0] == questions
        assert result["user_answers"] == ["React"]

    def test_no_llm_call(self):
        """alignment_collect must not invoke any LLM."""
        state = _base_state(clarifying_questions="Q1?")
        with patch("agent.nodes.interrupt", return_value="answer"):
            with patch("agent.nodes.alignment_llm") as mock_llm:
                from agent.nodes import alignment_collect
                alignment_collect(state)
                mock_llm.invoke.assert_not_called()


# ── design node ────────────────────────────────────────────────────────────────

class TestDesign:
    def test_returns_plan_draft(self):
        """Design node returns plan_draft string from the LLM."""
        mock_response = MagicMock()
        mock_response.content = "## Plan: React Auth App\n### TL;DR\nBuild it."
        state = _base_state(
            need_clarification=True,
            clarifying_questions="Q1: Auth?",
            user_answers=["JWT"],
        )

        with patch("agent.nodes.design_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            from agent.nodes import design
            result = design(state)

        assert result["plan_draft"] == "## Plan: React Auth App\n### TL;DR\nBuild it."

    def test_includes_research_summary_in_call(self):
        """Design LLM receives research_summary in the prompt."""
        mock_response = MagicMock()
        mock_response.content = "## Plan: ..."
        state = _base_state(user_answers=["JWT"], clarifying_questions="Q1?")

        with patch("agent.nodes.design_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            from agent.nodes import design
            design(state)

        call_msgs = mock_llm.invoke.call_args[0][0]
        human_msg = next(m for m in call_msgs if isinstance(m, HumanMessage))
        assert "React is a JS library" in human_msg.content

    def test_includes_user_answers_in_call(self):
        """Design LLM receives user_answers in the prompt."""
        mock_response = MagicMock()
        mock_response.content = "## Plan: ..."
        state = _base_state(user_answers=["JWT", "3 months"], clarifying_questions="Q1?")

        with patch("agent.nodes.design_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            from agent.nodes import design
            design(state)

        call_msgs = mock_llm.invoke.call_args[0][0]
        human_msg = next(m for m in call_msgs if isinstance(m, HumanMessage))
        assert "JWT" in human_msg.content


# ── present_plan node ─────────────────────────────────────────────────────────

class TestPresentPlan:
    def test_returns_ai_message(self):
        """present_plan returns a messages list containing one AIMessage."""
        from agent.nodes import _PLAN_READY, present_plan
        result = present_plan(_base_state(plan_draft="## Plan: Test"))
        assert "messages" in result
        msgs = result["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)
        assert msgs[0].content == _PLAN_READY


# ── refinement node ────────────────────────────────────────────────────────────

class TestRefinement:
    def _run_refinement(self, feedback_type: str, **fc_kwargs):
        """Helper to run refinement with a mocked LLM and interrupt."""
        fc = FeedbackClassification(feedback_type=feedback_type, **fc_kwargs)
        state = _base_state(
            plan_draft="## Plan: React Auth App\n### TL;DR\nBuild it.",
        )
        with patch("agent.nodes.interrupt", return_value="user feedback text"):
            with patch("agent.nodes.refinement_llm") as mock_llm:
                with patch("agent.nodes.save_plan", return_value="plans/plan_test.md"):
                    mock_llm.invoke.return_value = fc
                    from agent.nodes import refinement
                    return refinement(state)

    def test_approval_sets_approved_true(self):
        result = self._run_refinement("approval")
        assert result["approved"] is True
        assert result["final_plan"] is not None
        assert result["plan_file_path"] == "plans/plan_test.md"
        assert any("approved" in m.content.lower() or "saved" in m.content.lower()
                   for m in result.get("messages", []))

    def test_revision_updates_plan_draft(self):
        new_plan = "## Plan: Updated\n### TL;DR\nRevised."
        result = self._run_refinement("revision", updated_plan=new_plan)
        assert result["plan_draft"] == new_plan
        assert result["approved"] is False

    def test_question_adds_answer_message(self):
        result = self._run_refinement("question", answer="Because JWT is stateless.")
        assert result["approved"] is False
        msgs = result.get("messages", [])
        assert any("JWT" in m.content for m in msgs)

    def test_alternative_clears_plan(self):
        result = self._run_refinement("alternative")
        assert result["plan_draft"] is None
        assert result["approved"] is False
        assert result.get("research_summary") == ""

    def test_interrupt_called_with_plan_ready_message(self):
        """refinement must interrupt with the same _PLAN_READY string as present_plan."""
        from agent.nodes import _PLAN_READY
        fc = FeedbackClassification(feedback_type="approval")
        state = _base_state(plan_draft="## Plan: Test")

        with patch("agent.nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "approve"
            with patch("agent.nodes.refinement_llm") as mock_llm:
                with patch("agent.nodes.save_plan", return_value="plans/plan.md"):
                    mock_llm.invoke.return_value = fc
                    from agent.nodes import refinement
                    refinement(state)

        mock_interrupt.assert_called_once_with(_PLAN_READY)
