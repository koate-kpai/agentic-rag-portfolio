# test_agent.py — Tests for the WebScoutAgent class
#
# TEST STRATEGY: Mock the OpenAI client, test logic in isolation
# ---------------------------------------------------------------
# WebScoutAgent depends on AsyncOpenAI for LLM calls.  We mock the client
# to return controlled responses, testing only the agent's orchestration
# logic (tool execution, citation extraction) without network or API cost.
# The underlying search_web function is tested separately in test_tools.py.

import json
import pytest
from src.agent import WebScoutAgent
from src.models import Citation


@pytest.fixture
def agent():
    """
    Fixture that returns a WebScoutAgent constructed with a dummy API key.
    The dummy key is fine because we mock the underlying AsyncOpenAI client
    in each test that makes LLM calls.
    """
    return WebScoutAgent(api_key="sk-test-dummy-key")


class TestExecuteTool:
    """
    _execute_tool is a synchronous method that dispatches tool calls by name.
    Since search_web is mocked at the module level, no real search occurs.
    """

    def test_execute_tool_search(self, agent, mocker):
        """
        GIVEN tool name 'search_web' and valid JSON arguments,
        WHEN _execute_tool is called,
        THEN it calls search_web with the correct query and returns serialized results.
        """
        # Arrange: mock the underlying search_web function
        mock_search = mocker.patch("src.agent.search_web")
        mock_search.return_value = [
            {"title": "T", "snippet": "S", "url": "https://example.com"}
        ]

        # Act
        result = agent._execute_tool("search_web", '{"query": "test"}')

        # Assert
        mock_search.assert_called_once_with("test")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "T"

    def test_execute_tool_unknown(self, agent):
        """
        GIVEN an unknown tool name,
        WHEN _execute_tool is called,
        THEN a ValueError is raised.
        """
        with pytest.raises(ValueError, match="Unknown tool"):
            agent._execute_tool("nonexistent_tool", "{}")


class TestExtractCitations:
    """
    _extract_citations_from_results builds Citation objects directly from
    search tool output.  This is deliberately decoupled from the LLM's
    generated text to guarantee accurate citations regardless of what the
    model says.
    """

    def test_extract_citations_normal(self, agent):
        """
        GIVEN valid search results with URLs,
        WHEN _extract_citations_from_results is called,
        THEN it returns Citation objects with domain and url populated.
        """
        # Arrange: serialized search results with multiple entries
        tool_result = json.dumps(
            [
                {"title": "A", "snippet": "B", "url": "https://example.com/page1"},
                {"title": "C", "snippet": "D", "url": "https://other.org/page2"},
            ]
        )

        # Act
        citations = agent._extract_citations_from_results(tool_result)

        # Assert
        assert len(citations) == 2
        assert all(isinstance(c, Citation) for c in citations)
        assert citations[0].domain == "example.com"
        assert citations[0].url == "https://example.com/page1"
        assert citations[1].domain == "other.org"

    def test_extract_citations_empty(self, agent):
        """
        GIVEN empty search results,
        WHEN _extract_citations_from_results is called,
        THEN an empty list is returned.
        """
        tool_result = json.dumps([])
        citations = agent._extract_citations_from_results(tool_result)
        assert citations == []

    def test_extract_citations_missing_url(self, agent):
        """
        GIVEN a result entry without a 'url' key,
        WHEN _extract_citations_from_results is called,
        THEN that entry is silently skipped and remaining entries are returned.
        """
        tool_result = json.dumps(
            [
                {"title": "A", "snippet": "B", "url": "https://example.com/page1"},
                {"title": "C", "snippet": "D"},  # no url key
            ]
        )
        citations = agent._extract_citations_from_results(tool_result)
        assert len(citations) == 1
        assert citations[0].url == "https://example.com/page1"

    def test_extract_citations_empty_url(self, agent):
        """
        GIVEN a result entry with an empty url string,
        WHEN _extract_citations_from_results is called,
        THEN that entry is skipped (empty string is falsy).
        """
        tool_result = json.dumps(
            [
                {"title": "A", "snippet": "B", "url": ""},
            ]
        )
        citations = agent._extract_citations_from_results(tool_result)
        assert citations == []

    def test_extract_citations_invalid_json(self, agent):
        """
        GIVEN malformed JSON string,
        WHEN _extract_citations_from_results is called,
        THEN an empty list is returned (error is logged, not raised).
        """
        citations = agent._extract_citations_from_results("not valid json")
        assert citations == []

    def test_extract_citations_partial_failure(self, agent):
        """
        GIVEN a non-list JSON value (e.g., a dict),
        WHEN _extract_citations_from_results is called,
        THEN an empty list is returned gracefully.
        """
        citations = agent._extract_citations_from_results('{"title": "not a list"}')
        assert citations == []
