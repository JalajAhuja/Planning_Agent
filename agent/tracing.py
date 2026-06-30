"""LangSmith tracing helpers.

Tracing is activated purely via environment variables — no code changes needed.
Set the following in your .env file:

    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=lsv2_...
    LANGCHAIN_PROJECT=planning-agent   # optional, defaults to "default"

All graph invocations should use build_graph_config() to inject run metadata
so that LangSmith shows nested traces: Graph → Node → LLM → Tool.
"""

from __future__ import annotations

import datetime
import os
from typing import Any


def get_tracing_status() -> dict:
    """Return the current LangSmith tracing configuration."""
    tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key_set = bool(os.getenv("LANGCHAIN_API_KEY"))
    project = os.getenv("LANGCHAIN_PROJECT", "planning-agent")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    return {
        "active": tracing_on and api_key_set,
        "project": project,
        "endpoint": endpoint,
        "api_key_set": api_key_set,
        "tracing_on": tracing_on,
    }


def build_run_metadata(
    thread_id: str,
    session_id: str | None = None,
    topic: str | None = None,
    node: str | None = None,
) -> dict[str, Any]:
    """Build a metadata dict to attach to a LangSmith run.

    Injects thread_id, session_id, node, topic, and a UTC timestamp so every
    run in LangSmith is self-describing and can be filtered by session.
    """
    meta: dict[str, Any] = {
        "thread_id": thread_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "source": "planning-agent",
    }
    if session_id:
        meta["session_id"] = session_id
    if topic:
        meta["topic"] = topic[:120]  # keep it concise
    if node:
        meta["node"] = node
    return meta


def build_graph_config(
    thread_id: str,
    session_id: str | None = None,
    topic: str | None = None,
    extra_tags: list[str] | None = None,
) -> dict:
    """Return a LangGraph invocation config enriched with LangSmith metadata.

    The returned dict can be passed directly as the ``config`` argument to
    ``graph.invoke()`` or ``graph.stream()``.  It preserves the required
    ``configurable.thread_id`` key used by the SQLite checkpointer while also
    injecting ``run_name``, ``tags``, and ``metadata`` for LangSmith.

    Tags follow the convention: ["planning-agent", "streamlit", ...]
    Nested LLM runs inherit tags set via llms.with_config() in llms.py.
    """
    tags = ["planning-agent", "streamlit"]
    if extra_tags:
        tags.extend(extra_tags)

    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "planning-agent",
        "tags": tags,
        "metadata": build_run_metadata(
            thread_id=thread_id,
            session_id=session_id,
            topic=topic,
        ),
    }
