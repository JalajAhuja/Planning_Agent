from typing import Annotated, List, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

class PlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # full chat history, auto-appended
    phase: str                                             # Discovery | Alignment | Design | Refinement
    research_summary: Optional[str]                       # output of Discovery phase
    clarifying_questions: List[str]                       # questions generated in Alignment phase
    user_answers: List[str]                               # user responses to clarifying questions
    tool_outputs: List[str]                               # raw tool call results
    plan_draft: Optional[str]                             # working plan Markdown (Design phase)
    final_plan: Optional[str]                             # approved plan Markdown
    approved: Optional[bool]                              # True once user approves the plan


class FeedbackClassification(BaseModel):
    feedback_type: Literal["revision", "question", "alternative", "approval"]
    updated_plan: Optional[str] = None  # populated for "revision"
    answer: Optional[str] = None        # populated for "question"

