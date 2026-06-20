# tools.py

from duckduckgo_search import DDGS
from typing import List, Dict
from .logger import setup_logger

logger = setup_logger("webscout.tools")


def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Executes a live web search using DuckDuckGo.
    This function is synchronous because
    the underlying DDGS library is synchronous.
    When called from an async context,
    it should be wrapped in `asyncio.to_thread`.
    """
    logger.info(f"Executing web search: {query}")
    try:
        # Context manager ensures the DDGS session is closed properly
        with DDGS() as ddgs:
            results = []
            # Fetch text snippets from the search engine
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    }
                )
            logger.info(f"Search returned {len(results)} results")
            return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise RuntimeError(f"Web search failed: {str(e)}")
