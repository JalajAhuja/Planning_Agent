from pathlib import Path
import time


def save_plan(content: str) -> str:
    """
    Save the plan content to a Markdown file in the 'plans' directory.
    Returns the path to the saved file.
    """
    plans_dir = Path("plans")
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / f"plan_{time.strftime('%Y%m%d_%H%M%S')}.md"
    plan_path.write_text(content, encoding="utf-8")
    return str(plan_path)


# ── Plan template ─────────────────────────────────────────────────────────────
# Mirrors the exact structure demanded by design_prompt.
# Import this into prompts.py or inject it into the design prompt so the LLM
# always produces a consistently structured document.
PLAN_TEMPLATE = """\
## Plan: {title}

### TL;DR
{tldr}

### Steps

**Phase 1 — {phase_1_name}**
1. {step} — `{file_or_command}` *(parallel)*
2. {step}

**Phase 2 — {phase_2_name}**
3. {step}

### Relevant Files & References
- `path/to/file.py` — {reason}

### Verification
1. {command_or_check}
2. {command_or_check}

### Decisions
- **{decision_title}**: Chose X over Y because {reason}.

### Scope Exclusions
- {exclusion}

### Further Considerations
1. {consideration}
"""


def format_plan(
    title: str,
    tldr: str,
    steps: list[str],
    files: list[str],
    verification: list[str],
    decisions: list[str],
    exclusions: list[str],
    considerations: list[str],
) -> str:
    """
    Programmatically build a plan string from structured components.
    Useful when you want to assemble the plan from typed data rather than
    relying on raw LLM text.

    Each list item should be a plain string (no leading '- ' or numbering needed).
    """
    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- N/A"

    def _numbered(items: list[str]) -> str:
        return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items)) if items else "1. N/A"

    plan = f"## Plan: {title}\n\n"
    plan += f"### TL;DR\n{tldr}\n\n"
    plan += f"### Steps\n{_numbered(steps)}\n\n"
    plan += f"### Relevant Files & References\n{_bullets(files)}\n\n"
    plan += f"### Verification\n{_numbered(verification)}\n\n"
    plan += f"### Decisions\n{_bullets(decisions)}\n\n"
    plan += f"### Scope Exclusions\n{_bullets(exclusions)}\n\n"
    plan += f"### Further Considerations\n{_numbered(considerations)}\n"
    return plan