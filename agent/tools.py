import os
from pathlib import Path

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool


# ── Web Search (Tavily) ──────────────────────────────────────────────────────
tavily_search = TavilySearchResults(
    max_results=5,
    description=(
        "Search the web for current best practices, libraries, tutorials, and "
        "real-world examples relevant to the research topic."
    ),
)


# ── Wikipedia ────────────────────────────────────────────────────────────────
_wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=3000)
wikipedia = WikipediaQueryRun(
    api_wrapper=_wiki_wrapper,
    description=(
        "Look up foundational concepts, definitions, and background knowledge "
        "about a topic on Wikipedia."
    ),
)


# ── Local File Reader ─────────────────────────────────────────────────────────
_WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", ".")).resolve()


@tool
def read_file_tool(file_path: str) -> str:
    """Read the contents of a local file for code-aware planning.

    Args:
        file_path: Path to the file, relative to the workspace root.

    Returns:
        File contents as a string, or an error message if unreadable.
    """
    try:
        target = (_WORKSPACE_ROOT / file_path).resolve()
        # Prevent path traversal outside the workspace root
        target.relative_to(_WORKSPACE_ROOT)
        return target.read_text(encoding="utf-8")
    except ValueError:
        return "Error: Access denied — path is outside the workspace root."
    except FileNotFoundError:
        return f"Error: File not found — '{file_path}'"
    except OSError as exc:
        return f"Error reading file: {exc}"


# ── Tool registry (passed to LangGraph ToolNode) ─────────────────────────────
tools: list = [tavily_search, wikipedia, read_file_tool]
