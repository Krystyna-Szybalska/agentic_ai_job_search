import builtins
from unittest.mock import MagicMock
import pytest


@pytest.fixture
def mock_graph():
    return MagicMock()


def test_handle_remotive_confirmation_yes_returns_without_invoking(mock_graph, monkeypatch):
    from main import handle_remotive_confirmation

    state = MagicMock()
    state.values = {"remotive_search": "python"}

    monkeypatch.setattr(builtins, "input", lambda _: "yes")

    handle_remotive_confirmation(mock_graph, state, config={"configurable": {"thread_id": "t"}})

    # Resumption is owned by the dispatch loop in main(); the handler must NOT invoke.
    mock_graph.invoke.assert_not_called()


def test_handle_remotive_confirmation_no_exits(mock_graph, monkeypatch):
    from main import handle_remotive_confirmation

    state = MagicMock()
    state.values = {"remotive_search": "python"}

    monkeypatch.setattr(builtins, "input", lambda _: "no")

    with pytest.raises(SystemExit):
        handle_remotive_confirmation(mock_graph, state, config={"configurable": {"thread_id": "t"}})


def test_handle_remotive_confirmation_edit_then_yes(mock_graph, monkeypatch):
    from main import handle_remotive_confirmation

    state = MagicMock()
    state.values = {"remotive_search": "python"}

    inputs = iter(["AI engineer", "yes"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    # After update_state is called, the function re-fetches state via graph.get_state.
    # Mock that to return a state with the updated search query.
    updated_state = MagicMock()
    updated_state.values = {"remotive_search": "AI engineer"}
    mock_graph.get_state.return_value = updated_state

    handle_remotive_confirmation(mock_graph, state, config={"configurable": {"thread_id": "t"}})

    mock_graph.update_state.assert_any_call(
        {"configurable": {"thread_id": "t"}},
        {"remotive_search": "AI engineer"},
    )
    mock_graph.get_state.assert_called()
    # No direct invoke from the handler — dispatch loop resumes after return.
    mock_graph.invoke.assert_not_called()
