"""Tests for agent/state.py — Pydantic model validation."""
import pytest
from pydantic import ValidationError

from agent.state import AlignmentDecision, FeedbackClassification, PlannerState, RouterDecision


# ── RouterDecision ─────────────────────────────────────────────────────────────

def test_router_decision_valid():
    rd = RouterDecision(
        research_summary="Key facts about React.",
        need_clarification=True,
        clarifying_questions="Q1: What framework?",
    )
    assert rd.research_summary == "Key facts about React."
    assert rd.need_clarification is True
    assert rd.clarifying_questions == "Q1: What framework?"


def test_router_decision_no_questions_when_no_clarification():
    rd = RouterDecision(
        research_summary="Enough info.",
        need_clarification=False,
        clarifying_questions="",
    )
    assert rd.need_clarification is False
    assert rd.clarifying_questions == ""


def test_router_decision_missing_required_field():
    with pytest.raises(ValidationError):
        RouterDecision(need_clarification=True, clarifying_questions="Q1: ...")  # type: ignore[call-arg]


# ── AlignmentDecision ──────────────────────────────────────────────────────────

def test_alignment_decision_defaults():
    ad = AlignmentDecision(need_clarification=False)
    assert ad.need_clarification is False
    assert ad.clarifying_questions == ""


def test_alignment_decision_with_questions():
    ad = AlignmentDecision(
        need_clarification=True,
        clarifying_questions="**Q1: Timeline?**\n1. 1 month\n2. 3 months",
    )
    assert ad.need_clarification is True
    assert "Timeline" in ad.clarifying_questions


def test_alignment_decision_missing_required():
    with pytest.raises(ValidationError):
        AlignmentDecision()  # type: ignore[call-arg]


# ── FeedbackClassification ─────────────────────────────────────────────────────

def test_feedback_classification_approval():
    fc = FeedbackClassification(feedback_type="approval")
    assert fc.feedback_type == "approval"
    assert fc.updated_plan == ""
    assert fc.answer == ""
    assert fc.restart_discovery is False


def test_feedback_classification_revision():
    fc = FeedbackClassification(
        feedback_type="revision",
        updated_plan="## Plan: Updated\n### TL;DR\nNew approach.",
    )
    assert fc.feedback_type == "revision"
    assert "Updated" in fc.updated_plan


def test_feedback_classification_question():
    fc = FeedbackClassification(
        feedback_type="question",
        answer="The plan uses React because it is widely supported.",
    )
    assert fc.feedback_type == "question"
    assert "React" in fc.answer


def test_feedback_classification_alternative():
    fc = FeedbackClassification(feedback_type="alternative", restart_discovery=True)
    assert fc.feedback_type == "alternative"
    assert fc.restart_discovery is True


def test_feedback_classification_missing_required():
    with pytest.raises(ValidationError):
        FeedbackClassification()  # type: ignore[call-arg]
