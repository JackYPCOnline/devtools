"""Tests for the bug-verifier write guard in agent_runner."""
from unittest.mock import patch

import pytest

from .. import agent_runner


def test_guard_passes_when_label_tool_invoked_this_run():
    """A run that invoked add_issue_labels satisfies the guard."""
    with patch.object(agent_runner, "get_invoked_write_tools", lambda: ["add_issue_labels"]):
        agent_runner._enforce_required_writes("bug-verifier-3216")  # must not raise


def test_guard_passes_when_comment_tool_invoked_this_run():
    """A comment on the issue also satisfies the guard."""
    with patch.object(agent_runner, "get_invoked_write_tools", lambda: ["add_issue_comment"]):
        agent_runner._enforce_required_writes("bug-verifier-3216")  # must not raise


def test_guard_passes_on_resume_when_issue_already_has_triage_label():
    """A resumed run that writes nothing is accepted if a prior triage label landed."""
    with patch.object(agent_runner, "get_invoked_write_tools", lambda: []), \
         patch.object(agent_runner, "get_issue_label_names",
                      lambda *a, **k: ["bug", "bug-needs-info"]):
        agent_runner._enforce_required_writes("bug-verifier-3216")  # must not raise


def test_guard_fails_when_no_write_and_no_existing_triage_label():
    """The silent no-op: no issue-facing write and no prior triage label -> failure."""
    with patch.object(agent_runner, "get_invoked_write_tools", lambda: []), \
         patch.object(agent_runner, "get_issue_label_names",
                      lambda *a, **k: ["bug", "area-otel"]):
        with pytest.raises(RuntimeError):
            agent_runner._enforce_required_writes("bug-verifier-3216")


def test_guard_ignores_non_bug_verifier_modes():
    """Other modes have no mandatory-write guarantee and are never failed here."""
    with patch.object(agent_runner, "get_invoked_write_tools", lambda: []):
        agent_runner._enforce_required_writes("implementer-my-branch")
        agent_runner._enforce_required_writes("reviewer-42")
        agent_runner._enforce_required_writes(None)


def test_issue_number_parsed_from_session_id():
    assert agent_runner._issue_number_from_session_id("bug-verifier-3216") == 3216
    assert agent_runner._issue_number_from_session_id("bug-verifier-abc") is None
