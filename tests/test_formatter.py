"""Tests for agent/formatter.py"""
import re
from pathlib import Path

import pytest


class TestSavePlan:
    def test_creates_file_in_plans_dir(self, tmp_path, monkeypatch):
        """save_plan writes to a plans/ subdirectory."""
        monkeypatch.chdir(tmp_path)
        from agent.formatter import save_plan

        path = save_plan("## Plan: Test\n### TL;DR\nTest plan.")
        assert Path(path).exists()
        assert "plans" in path

    def test_file_contains_content(self, tmp_path, monkeypatch):
        """save_plan writes the exact content to the file."""
        monkeypatch.chdir(tmp_path)
        from agent.formatter import save_plan

        content = "## Plan: React App\n### TL;DR\nBuild it."
        path = save_plan(content)
        assert Path(path).read_text(encoding="utf-8") == content

    def test_filename_has_timestamp(self, tmp_path, monkeypatch):
        """Filename contains a timestamp in YYYYMMDD_HHMMSS format."""
        monkeypatch.chdir(tmp_path)
        from agent.formatter import save_plan

        path = save_plan("content")
        filename = Path(path).name
        assert re.search(r"plan_\d{8}_\d{6}\.md", filename), f"Unexpected filename: {filename}"

    def test_returns_string_path(self, tmp_path, monkeypatch):
        """save_plan returns a string, not a Path object."""
        monkeypatch.chdir(tmp_path)
        from agent.formatter import save_plan

        result = save_plan("content")
        assert isinstance(result, str)

    def test_creates_plans_dir_if_missing(self, tmp_path, monkeypatch):
        """save_plan creates the plans/ directory if it does not exist."""
        monkeypatch.chdir(tmp_path)
        plans_dir = tmp_path / "plans"
        assert not plans_dir.exists()
        from agent.formatter import save_plan

        save_plan("content")
        assert plans_dir.is_dir()


class TestFormatPlan:
    def test_basic_structure(self):
        """format_plan returns a string with all required section headers."""
        from agent.formatter import format_plan

        result = format_plan(
            title="My Project",
            tldr="Build a thing.",
            steps=["Step one", "Step two"],
            files=["src/app.py — main entry"],
            verification=["Run pytest"],
            decisions=["Chose React over Vue"],
            exclusions=["Mobile app out of scope"],
            considerations=["Add CI later"],
        )
        assert "## Plan: My Project" in result
        assert "### TL;DR" in result
        assert "### Steps" in result
        assert "### Relevant Files" in result
        assert "### Verification" in result
        assert "### Decisions" in result
        assert "### Scope Exclusions" in result
        assert "### Further Considerations" in result

    def test_steps_numbered(self):
        """Steps are numbered in the output."""
        from agent.formatter import format_plan

        result = format_plan(
            title="T", tldr="T", steps=["Alpha", "Beta"],
            files=[], verification=[], decisions=[], exclusions=[], considerations=[],
        )
        assert "1. Alpha" in result
        assert "2. Beta" in result

    def test_files_bulleted(self):
        """Files are formatted as bullet points."""
        from agent.formatter import format_plan

        result = format_plan(
            title="T", tldr="T", steps=[],
            files=["app.py — main", "utils.py — helpers"],
            verification=[], decisions=[], exclusions=[], considerations=[],
        )
        assert "- app.py — main" in result
        assert "- utils.py — helpers" in result

    def test_empty_lists_show_na(self):
        """Empty lists produce a '- N/A' or '1. N/A' placeholder."""
        from agent.formatter import format_plan

        result = format_plan(
            title="T", tldr="T", steps=[], files=[], verification=[],
            decisions=[], exclusions=[], considerations=[],
        )
        assert "N/A" in result
