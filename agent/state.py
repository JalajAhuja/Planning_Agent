from typing import Annotated
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class PlannerState(TypedDict):
    # Entire conversation -- single source of truth
    messages: Annotated[list[BaseMessage], add_messages]

    # Discovery -> ResearchSummary
    research_summary: NotRequired[str]

    # Alignment
    need_clarification: NotRequired[bool]
    clarifying_questions: NotRequired[str]
    user_answers: NotRequired[list[str]]

    # Design
    plan_draft: NotRequired[str | None]

    # Refinement
    approved: NotRequired[bool]

    # Final output
    final_plan: NotRequired[str | None]

    # Path to the saved markdown file (set on approval)
    plan_file_path: NotRequired[str | None]


class RouterDecision(BaseModel):
    research_summary: str = Field(
        description="Condensed research from Discovery."
    )
    need_clarification: bool = Field(
        description="Whether user clarification is required."
    )
    clarifying_questions: str = Field(
        description="Markdown containing every clarification question. Empty string if not needed."
    )


class AlignmentDecision(BaseModel):
    need_clarification: bool = Field(
        description="Whether additional information is required from the user."
    )
    clarifying_questions: str = Field(
        default="",
        description=(
            "Questions to ask the user if clarification is required. "
            "Leave empty when need_clarification=False."
        ),
    )


class FeedbackClassification(BaseModel):
    feedback_type: str = Field(
        description="One of: approval, revision, question, alternative."
    )
    updated_plan: str = Field(
        default="",
        description="The complete updated plan text. Only populated for revision.",
    )
    answer: str = Field(
        default="",
        description="Answer to user question. Only populated for question.",
    )
    restart_discovery: bool = Field(
        default=False,
        description="Set to True when feedback_type is alternative.",
    )
