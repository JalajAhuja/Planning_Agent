from __future__ import annotations

import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import (
    alignment_decide,
    alignment_collect,
    design,
    discovery,
    present_plan,
    refinement,
    research_summary,
)
from agent.routers import route_after_discovery, route_after_alignment_decide, route_after_refinement
from agent.state import PlannerState
from agent.tools import tools

load_dotenv()


def build_graph():
    builder = StateGraph(PlannerState)

    # -- Nodes -----------------------------------------------------------------
    builder.add_node("discovery", discovery)
    builder.add_node("tool_executor", ToolNode(tools, handle_tool_errors=True))
    builder.add_node("research_summary", research_summary)
    builder.add_node("alignment_decide", alignment_decide)
    builder.add_node("alignment_collect", alignment_collect)
    builder.add_node("design", design)
    builder.add_node("present_plan", present_plan)
    builder.add_node("refinement", refinement)

    # -- Edges -----------------------------------------------------------------
    builder.add_edge(START, "discovery")

    # Discovery ReAct loop -> ResearchSummary when done
    builder.add_conditional_edges(
        "discovery",
        route_after_discovery,
        {"tool_executor": "tool_executor", "research_summary": "research_summary"},
    )
    builder.add_edge("tool_executor", "discovery")

    builder.add_edge("research_summary", "alignment_decide")

    # alignment_decide routes to alignment_collect (if questions needed) or design
    builder.add_conditional_edges(
        "alignment_decide",
        route_after_alignment_decide,
        {"alignment_collect": "alignment_collect", "design": "design"},
    )
    builder.add_edge("alignment_collect", "design")

    builder.add_edge("design", "present_plan")
    builder.add_edge("present_plan", "refinement")

    builder.add_conditional_edges(
        "refinement",
        route_after_refinement,
        {
            "present_plan": "present_plan",
            "discovery": "discovery",
            "__end__": END,
        },
    )

    # -- Checkpointer ----------------------------------------------------------
    db_path = Path(__file__).parent.parent / "checkpoints.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
