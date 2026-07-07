# test_tools.py — Tests for the DuckDuckGo search wrapper
#
# TEST STRATEGY: Mock external HTTP calls
# ----------------------------------------
# DuckDuckGo's DDGS class makes real HTTP requests.  We mock DDGS at the
# class level so tests are fast, deterministic, and require no network or
# API keys.  The mock returns synthetic results matching the real schema.

from unittest.mock import patch, MagicMock
import pytest
from src.tools import search_web


@patch("src.tools.DDGS")
def test_search_web_success(mock_ddgs_class):
    """
    GIVEN a query and a mocked DuckDuckGo that returns 2 results,
    WHEN search_web is called,
    THEN it returns a list of dicts with the expected keys (title, snippet, url)
    and the count matches.
    """
    # Arrange: configure the mock DDGS context manager to yield fake results
    mock_instance = MagicMock()
    mock_instance.text.return_value = [
        {"title": "Result 1", "body": "Snippet 1", "href": "https://example.com/1"},
        {"title": "Result 2", "body": "Snippet 2", "href": "https://example.com/2"},
    ]
    mock_ddgs_class.return_value.__enter__.return_value = mock_instance

    # Act
    results = search_web("test query", max_results=2)

    # Assert
    assert len(results) == 2
    assert results[0]["title"] == "Result 1"
    assert results[0]["snippet"] == "Snippet 1"
    assert results[0]["url"] == "https://example.com/1"
    assert results[1]["title"] == "Result 2"

    # Verify the mock was called correctly
    mock_instance.text.assert_called_once_with("test query", max_results=2)


@patch("src.tools.DDGS")
def test_search_web_empty_results(mock_ddgs_class):
    """
    GIVEN a query that returns no results,
    WHEN search_web is called,
    THEN it returns an empty list (not None, not an error).
    """
    # Arrange
    mock_instance = MagicMock()
    mock_instance.text.return_value = []
    mock_ddgs_class.return_value.__enter__.return_value = mock_instance

    # Act
    results = search_web("nonexistent query")

    # Assert
    assert results == []


@patch("src.tools.DDGS")
def test_search_web_failure(mock_ddgs_class):
    """
    GIVEN a DuckDuckGo that raises an exception,
    WHEN search_web is called,
    THEN a RuntimeError is raised with a descriptive message.
    """
    # Arrange: the mock raises when entering the context manager
    mock_instance = MagicMock()
    mock_instance.text.side_effect = Exception("Connection timeout")
    mock_ddgs_class.return_value.__enter__.return_value = mock_instance

    # Act & Assert
    with pytest.raises(RuntimeError, match="Web search failed"):
        search_web("failing query")
