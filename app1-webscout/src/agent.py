# agent.py

import json
import asyncio
from typing import AsyncGenerator, Dict, List
from openai import AsyncOpenAI
from .logger import setup_logger
from .tools import search_web
from .models import Citation

logger = setup_logger("webscout.agent")

SYSTEM_PROMPT = """You are a precise search assistant. Your goal is to provide concise, factual answers (maximum 3 sentences) based on web search results when needed. You have access to a `search_web` function. If the user's query is a casual greeting or does not require up-to-date information, answer directly without using the function. Otherwise, use the function to get context and synthesize an answer. Always include citations from the search results in your final response."""

FUNCTION_DEFINITION = {
    "name": "search_web",
    "description": "Search the web for current information.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    },
}


class WebScoutAgent:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    def _execute_tool(self, name: str, arguments: str) -> str:
        """
        Synchronous tool runner. Must be synchronous so it can be safely
        offloaded to a thread with `asyncio.to_thread`.
        """
        if name == "search_web":
            args = json.loads(arguments)
            results = search_web(args["query"])
            return json.dumps(results)  # serialized list of {title, snippet, url}
        raise ValueError(f"Unknown tool: {name}")

    def _extract_citations_from_results(self, tool_result_json: str) -> List[Citation]:
        """
        Parse the serialized search results and build reliable Citation objects.
        This ensures we always return valid citations, regardless of what the LLM
        includes in its generated text.
        """
        citations = []
        try:
            results = json.loads(tool_result_json)
            for item in results:
                url = item.get("url", "")
                if url:
                    from urllib.parse import urlparse

                    parsed = urlparse(url)
                    citations.append(Citation(domain=parsed.netloc, url=url))
        except Exception as e:
            logger.error(f"Failed to extract citations from tool result: {e}")
        return citations

    async def answer_stream(self, query: str) -> AsyncGenerator[str, None]:
        """Agentic loop with streaming answer + guaranteed citations."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        # Phase 1: Decide tool usage
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=[{"type": "function", "function": FUNCTION_DEFINITION}],
                tool_choice="auto",
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            yield f"data: {json.dumps({'error': 'Failed to process query'})}\n\n"
            return

        msg = response.choices[0].message
        search_citations: List[Citation] = []  # will hold accurate citations

        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            # Offload synchronous search to a thread (fixes the earlier bug)
            try:
                tool_result = await asyncio.to_thread(
                    self._execute_tool, tool_name, tool_args
                )
            except Exception as e:
                logger.error(f"Search execution error: {e}")
                yield f"data: {json.dumps({'error': 'Search execution failed'})}\n\n"
                return

            # Build citations directly from the search results
            search_citations = self._extract_citations_from_results(tool_result)

            # Add tool result to conversation history
            messages.append(msg)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result}
            )

        final_answer = ""

        # Phase 2: Stream final synthesis
        try:
            stream = await self.client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, stream=True, temperature=0.0
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    final_answer += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # Final event: complete answer + guaranteed citations
            final_event = {
                "answer": final_answer.strip(),
                "citations": [c.model_dump() for c in search_citations],
            }
            yield f"data: {json.dumps(final_event)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"data: {json.dumps({'error': 'Stream generation failed'})}\n\n"
