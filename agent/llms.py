from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from agent.state import (
    RouterDecision,
    AlignmentDecision,
    FeedbackClassification,
)

from agent.tools import tools

load_dotenv()

# ---------------------------------------------------------
# Base Models
# ---------------------------------------------------------

FAST_MODEL = "gpt-4o"
SMART_MODEL = "gpt-5.2"

# ---------------------------------------------------------
# Discovery
# ---------------------------------------------------------

# This model ONLY decides which tools to call.
# run_name and tags surface in LangSmith as nested children of the graph run.
discovery_tool_llm = (
    ChatOpenAI(
        model=FAST_MODEL,
        temperature=0,
    )
    .bind_tools(tools)
    .with_config(run_name="discovery-tools", tags=["planning-agent", "discovery"])
)

# This model ONLY converts gathered research into structured output.
discovery_summary_llm = (
    ChatOpenAI(
        model=FAST_MODEL,
        temperature=0,
    )
    .with_structured_output(RouterDecision)
    .with_config(run_name="discovery-summary", tags=["planning-agent", "discovery"])
)

# ---------------------------------------------------------
# Alignment
# ---------------------------------------------------------

alignment_llm = (
    ChatOpenAI(
        model=FAST_MODEL,
        temperature=0,
    )
    .with_structured_output(AlignmentDecision)
    .with_config(run_name="alignment", tags=["planning-agent", "alignment"])
)

# ---------------------------------------------------------
# Design
# ---------------------------------------------------------

design_llm = (
    ChatOpenAI(
        model=SMART_MODEL,
        temperature=0.3,
    )
    .with_config(run_name="design", tags=["planning-agent", "design"])
)

# ---------------------------------------------------------
# Refinement
# ---------------------------------------------------------

refinement_llm = (
    ChatOpenAI(
        model=FAST_MODEL,
        temperature=0,
    )
    .with_structured_output(FeedbackClassification)
    .with_config(run_name="refinement", tags=["planning-agent", "refinement"])
)