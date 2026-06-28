from typing import Annotated, List, Optional, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class PlannerState(TypedDict):
    topic: str
    messages: Annotated[List[BaseMessage], add_messages]  # full chat history, auto-appended
    phase: str                                             # discovery | alignment | design | refinement | complete
    research_summary: Optional[str]                       # output of Discovery phase
    clarifying_questions: List[str]                       # questions generated in Alignment phase
    user_answers: List[str]                               # user responses to clarifying questions
    tool_outputs: List[str]                               # raw tool call results
    plan_draft: Optional[str]                             # working plan Markdown (Design phase)
    final_plan: Optional[str]                             # approved plan Markdown
    approved: Optional[bool]                              # True once user approves the plan
    refinement_action: Optional[str]                      # approve | revise | question | alternative
    display_content: Optional[str]                        # content shown by present_plan node

class RouterDecision(BaseModel):
    need_clarification: bool
    clarifying_questions: List[str]
    user_answers: List[str]
    max_questions: int

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what reader should do/understand")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120-550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_code: bool = False

class FeedbackClassification(BaseModel):
    feedback_type: Literal["revision", "question", "alternative", "approval"]
    updated_plan: Optional[str] = None  # populated for "revision"
    answer: Optional[str] = None        # populated for "question"

