from typing import Literal, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


from agent.state import PlannerState, FeedbackClassification
from agent.tools import tools
from agent.prompts import discovery_prompt, alignment_prompt, design_prompt, refinement_prompt
from agent.formatter import save_plan

load_dotenv()




# ── LLM setup (per-phase temperatures) ──────────────────────────────────────
_discovery_llm = ChatOpenAI(model="gpt-5.1", temperature=0)
_alignment_llm = ChatOpenAI(model="gpt-5.1", temperature=0.2)
_design_llm = ChatOpenAI(model="gpt-5.1", temperature=0.4)
_refinement_base_llm = ChatOpenAI(model="gpt-5.1", temperature=0.2)

_llm_with_tools = _discovery_llm.bind_tools(tools)                              # discovery: tool-calling
_refinement_llm = _refinement_base_llm.with_structured_output(FeedbackClassification)  # refinement: classification


# ── Node: discovery ───────────────────────────────────────────────────────────
def discovery(state: PlannerState) -> dict:
    """Phase 1: Research the topic using all available tools before planning."""
    messages = list(state["messages"])

    # Inject the discovery system prompt at position 0 if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=discovery_prompt)] + messages

    response = _llm_with_tools.invoke(messages)

    updates: dict = {"messages": [response]}

    # Once the LLM stops emitting tool calls, its final message IS the research summary
    if not response.tool_calls:
        updates["research_summary"] = response.content
        updates["phase"] = "Alignment"

    return updates


# ── Node: tool_executor (ToolNode handles all tool calls from discovery) ──────
tool_executor = ToolNode(tools)


# ── Node: alignment ───────────────────────────────────────────────────────────
def alignment(state: PlannerState) -> dict:
    """Phase 2: Clarify any ambiguities from the research summary before planning."""
    # Build a focused prompt using just the research summary — no full message history
    messages = [
        SystemMessage(content=alignment_prompt),
        HumanMessage(content=f"Research summary:\n\n{state.get('research_summary', '')}"),
    ]

    response = _alignment_llm.invoke(messages)
    questions_text = response.content

    # Pause the graph and surface the questions to the caller (e.g. Streamlit).
    # When the caller resumes via graph.invoke(Command(resume=answer), config),
    # interrupt() returns that answer value.
    user_input: str = interrupt(questions_text)

    return {
        "messages": [response],
        "clarifying_questions": [questions_text],
        "user_answers": [user_input],
        "phase": "Design",
    }


# ── Node: design ──────────────────────────────────────────────────────────────
def design(state: PlannerState) -> dict:
    """Phase 3: Produce a complete, structured, actionable plan based on the research and clarifications."""
    messages = [
        SystemMessage(content=design_prompt),
        HumanMessage(content=f"Research summary:\n\n{state.get('research_summary', '')}"),
        HumanMessage(content=f"User answers:\n\n{state.get('user_answers', [])}"),
    ]

    response = _design_llm.invoke(messages)
    plan_draft = response.content

    # Save plan to disk
    save_plan(plan_draft)

    # Pause and surface the plan to the user; resume value is their feedback
    user_input: str = interrupt(plan_draft)

    return {
        "messages": [response, HumanMessage(content=user_input)],  # feedback goes into messages for refinement to read
        "plan_draft": plan_draft,
        "phase": "Refinement",
    }


# ── Node: refinement ─────────────────────────────────────────────────────────
def refinement(state: PlannerState) -> Command:
    """Phase 4: Classify user feedback and route — revise plan, answer question,
    restart discovery, or approve and end."""
    messages = [
        SystemMessage(content=refinement_prompt),
        HumanMessage(content=f"Current plan:\n\n{state.get('plan_draft', '')}"),
        HumanMessage(content=f"User feedback:\n\n{state['messages'][-1].content}"),
    ]

    result: FeedbackClassification = _refinement_llm.invoke(messages)

    if result.feedback_type == "approval":
        return Command(
            goto=END,
            update={
                "approved": True,
                "final_plan": state.get("plan_draft"),
                "phase": "Complete",
            },
        )

    if result.feedback_type == "alternative":
        # Surface a confirmation message, then restart research with new direction
        user_input: str = interrupt(
            "Understood — I'll restart research with your new direction. "
            "Please confirm or add any details."
        )
        return Command(
            goto="discovery",
            update={
                "messages": [HumanMessage(content=user_input)],
                "phase": "Discovery",
                "research_summary": None,
                "plan_draft": None,
            },
        )

    if result.feedback_type == "revision":
        updated_plan = result.updated_plan or state.get("plan_draft", "")
        # Save revised plan to disk
        save_plan(updated_plan)
        # Surface revised plan; resume value is next round of feedback
        user_input = interrupt(updated_plan)
        return Command(
            goto="refinement",
            update={
                "messages": [HumanMessage(content=user_input)],
                "plan_draft": updated_plan,
                "phase": "Refinement",
            },
        )

    # "question" — answer inline, wait for follow-up
    user_input = interrupt(result.answer or "")
    return Command(
        goto="refinement",
        update={"messages": [HumanMessage(content=user_input)]},
    )


# ── Graph assembly ────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(PlannerState)

    builder.add_node("discovery", discovery)
    builder.add_node("tool_executor", tool_executor)
    builder.add_node("alignment", alignment)
    builder.add_node("design", design)
    builder.add_node("refinement", refinement)

    # Discovery ReAct loop: LLM calls tools → tools feed back → LLM decides to stop
    builder.add_edge(START, "discovery")
    builder.add_conditional_edges(
        "discovery",
        tools_condition,                          # built-in: checks for tool_calls on last message
        {"tools": "tool_executor", END: "alignment"},
    )
    builder.add_edge("tool_executor", "discovery")  # loop back with tool results

    # Linear flow after discovery
    builder.add_edge("alignment", "design")
    builder.add_edge("design", "refinement")
    # refinement uses Command(goto=...) for dynamic routing — no static edge needed

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()

