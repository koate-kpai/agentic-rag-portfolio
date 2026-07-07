# test_pipeline.py — Tests for the DocuMindRAG pipeline
#
# TEST STRATEGY: Mock AsyncOpenAI, test orchestration logic
# -----------------------------------------------------------
# DocuMindRAG makes three distinct LLM calls: classify, evaluate, and generate.
# We mock the AsyncOpenAI client's chat.completions.create method at the
# class level to return controlled responses.  This keeps tests fast,
# deterministic, and free of API costs.  The vector store is real (see
# test_vectorstore.py) so retrieval logic is exercised naturally.
#
# Each test follows the Arrange-Act-Assert (AAA) pattern:
#   Arrange — set up mocks and inputs
#   Act     — call the method under test
#   Assert  — verify outputs match expectations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.rag_pipeline import DocuMindRAG, VALID_DOMAINS


# ---------------------------------------------------------------------------
# Helper: build a mocked OpenAI chat completion response for tool calls
# ---------------------------------------------------------------------------
def _make_tool_call_response(tool_name: str, arguments: dict, finish_reason: str = "stop"):
    """
    Returns a MagicMock that mimics the OpenAI chat.completions.create response
    structure for a function/tool call.  This mirrors:
      response.choices[0].message.tool_calls[0].function.{name,arguments}
    """
    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.tool_calls = [
        MagicMock(
            function=MagicMock(
                name=tool_name,
                arguments=json.dumps(arguments),
            )
        )
    ]
    mock_choice.message.content = None  # No text response when tool is called

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _make_text_response(text: str, finish_reason: str = "stop"):
    """
    Returns a MagicMock for a plain-text (non-tool) response.
    """
    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = text
    mock_choice.message.tool_calls = None  # No tool call

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rag():
    """
    Fixture returning a DocuMindRAG instance with a dummy API key.
    The AsyncOpenAI client is mocked per-test via mocker.patch.
    """
    return DocuMindRAG(api_key="sk-test-dummy-key")


# ---------------------------------------------------------------------------
# Domain classification tests
# ---------------------------------------------------------------------------

class TestClassifyDomain:
    """
    _classify_domain uses function calling to map a query to one of the
    three supported domains (finance, healthcare, legal) or "unknown".
    """

    @pytest.mark.asyncio
    async def test_classify_finance(self, rag, mocker):
        """
        GIVEN a finance-related query,
        WHEN _classify_domain is called,
        THEN it returns "finance".
        """
        # Arrange
        mock_create = AsyncMock(return_value=_make_tool_call_response(
            "classify_domain", {"domain": "finance"}
        ))
        mocker.patch.object(rag.client.chat.completions, "create", mock_create)

        # Act
        domain = await rag._classify_domain("What are SEC rules?")

        # Assert
        assert domain == "finance"

    @pytest.mark.asyncio
    async def test_classify_healthcare(self, rag, mocker):
        """
        GIVEN a healthcare-related query,
        WHEN _classify_domain is called,
        THEN it returns "healthcare".
        """
        mock_create = AsyncMock(return_value=_make_tool_call_response(
            "classify_domain", {"domain": "healthcare"}
        ))
        mocker.patch.object(rag.client.chat.completions, "create", mock_create)

        domain = await rag._classify_domain("HIPAA regulations")

        assert domain == "healthcare"

    @pytest.mark.asyncio
    async def test_classify_legal(self, rag, mocker):
        """
        GIVEN a legal-related query,
        WHEN _classify_domain is called,
        THEN it returns "legal".
        """
        mock_create = AsyncMock(return_value=_make_tool_call_response(
            "classify_domain", {"domain": "legal"}
        ))
        mocker.patch.object(rag.client.chat.completions, "create", mock_create)

        domain = await rag._classify_domain("Contract law principles")

        assert domain == "legal"

    @pytest.mark.asyncio
    async def test_classify_no_tool_call_returns_unknown(self, rag, mocker):
        """
        GIVEN a model response with no tool_calls (edge case),
        WHEN _classify_domain is called,
        THEN it returns "unknown".
        """
        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.tool_calls = None
        mock_choice.message.content = "I am not sure"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        mocker.patch.object(rag.client.chat.completions, "create", mock_create)

        domain = await rag._classify_domain("Some ambiguous question")

        assert domain == "unknown"

    @pytest.mark.asyncio
    async def test_classify_invalid_domain_in_tool_call(self, rag, mocker):
        """
        GIVEN a tool call returning an invalid domain (e.g. "astrophysics"),
        WHEN _classify_domain is called,
        THEN it returns "unknown" (not in VALID_DOMAINS).
        """
        mock_create = AsyncMock(return_value=_make_tool_call_response(
            "classify_domain", {"domain": "astrophysics"}
        ))
        mocker.patch.object(rag.client.chat.completions, "create", mock_create)

        domain = await rag._classify_domain("Black hole physics")

        assert domain == "unknown"

    @pytest.mark.asyncio
    async def test_classify_malformed_json_returns_unknown(self, rag, mocker):
        """
        GIVEN a tool call with non-JSON arguments,
        WHEN _classify_domain is called,
        THEN it returns "unknown" (JSONDecodeError caught).
        """
        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.tool_calls = [
            MagicMock(
                function=MagicMock(
                    name="classify_domain",
                    arguments="not valid json",  # will raise JSONDecodeError
                )
            )
        ]
        mock_choice.message.content = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        mocker.patch.object(rag.client.chat.completions, "create", mock_create)

        domain = await rag._classify_domain("Query with bad JSON")

        assert domain == "unknown"


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------

class TestProcessQuery:
    """
    These tests exercise process_query end-to-end with mocked LLM calls
    but a real vector store.  We mock only the three LLM interactions:
    classify, evaluate, and generate.
    """

    @pytest.mark.asyncio
    async def test_process_query_finance_happy_path(self, rag, mocker):
        """
        GIVEN a finance query and sufficient retrieval on first attempt,
        WHEN process_query is called,
        THEN it returns an answer, domain="finance", and retrieved docs.
        """
        # Arrange: mock all three LLM calls in a single AsyncMock with
        # side_effect as a list of return values.  Each call to
        # `await client.chat.completions.create(...)` returns the next value
        # from the list, wrapped in a coroutine by AsyncMock.
        mock_create = AsyncMock(side_effect=[
            _make_tool_call_response("classify_domain", {"domain": "finance"}),
            _make_text_response("yes"),
            _make_text_response("SEC Rule 10b-5 prohibits securities fraud."),
        ])
        mocker.patch.object(rag.client.chat.completions, "create", mock_create)

        # Act
        result = await rag.process_query("What is SEC Rule 10b-5?")

        # Assert
        assert result["domain"] == "finance"
        assert "SEC Rule 10b-5" in result["answer"]
        assert len(result["retrieved_docs"]) > 0
        assert all(d["doc_id"].startswith("fin-") for d in result["retrieved_docs"])
        assert "fin-001" in result["citations"]

    @pytest.mark.asyncio
    async def test_process_query_unknown_domain_early_exit(self, rag, mocker):
        """
        GIVEN a query that cannot be classified into a known domain,
        WHEN process_query is called,
        THEN it returns early with a clear message and no retrieval.
        """
        # Arrange: a single AsyncMock for the one LLM call (classification)
        mock_create = AsyncMock(side_effect=[
            _make_tool_call_response("classify_domain", {"domain": "unknown"}),
        ])
        mocker.patch.object(rag.client.chat.completions, "create", mock_create)

        # Act
        result = await rag.process_query("What is the meaning of life?")

        # Assert
        assert result["domain"] == "unknown"
        assert "couldn't classify" in result["answer"].lower()
        assert result["retrieved_docs"] == []
        assert result["citations"] == []

    @pytest.mark.asyncio
    async def test_process_query_with_reformulation(self, rag, mocker):
        """
        GIVEN a query where initial retrieval is insufficient,
        WHEN the evaluator suggests a reformulated query,
        THEN the pipeline re-retrieves with the reformulated query and answers.
        """
        # Arrange: four LLM calls: classify, evaluate (no), evaluate (yes), generate
        mock_create = AsyncMock(side_effect=[
            _make_tool_call_response("classify_domain", {"domain": "legal"}),
            _make_text_response(
                "no\nReformulated query: contract consideration offer acceptance"
            ),
            _make_text_response("yes"),
            _make_text_response(
                "A contract requires offer, acceptance, and consideration."
            ),
        ])
        mocker.patch.object(rag.client.chat.completions, "create", mock_create)

        # Act
        result = await rag.process_query("Tell me about contracts")

        # Assert
        assert result["domain"] == "legal"
        assert "offer" in result["answer"].lower()
        # Thought process should include reformulation
        thought = " ".join(result["thought_process"]).lower()
        assert "reformulating" in thought

    @pytest.mark.asyncio
    async def test_process_query_identity_guard_prevents_loop(self, rag, mocker):
        """
        GIVEN an evaluator that returns 'no' with the same query as reformulation,
        WHEN process_query is called,
        THEN the pipeline breaks early (identity guard) instead of looping.
        """
        # Arrange: three LLM calls (classify + evaluate + generate).
        # The identity guard breaks the self-correction loop after the
        # first evaluate, but the pipeline still runs _generate_answer
        # with the (unchanged) retrieved docs.
        mock_create = AsyncMock(side_effect=[
            _make_tool_call_response("classify_domain", {"domain": "legal"}),
            _make_text_response("no\nReformulated query: contracts"),
            _make_text_response("A contract requires offer, acceptance, and consideration."),
        ])
        mocker.patch.object(rag.client.chat.completions, "create", mock_create)

        # Act
        result = await rag.process_query("contracts")

        # Assert
        # Should have thought process showing the guard triggered
        thought = " ".join(result["thought_process"]).lower()
        assert "unchanged" in thought or "stopping" in thought
