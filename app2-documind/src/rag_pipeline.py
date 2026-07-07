# rag_pipeline.py — Agentic RAG pipeline for app2-documind
#
# ARCHITECTURAL DECISION: Function calling for domain classification
# -------------------------------------------------------------------
# Prompt-based classification (asking GPT to "return one of: X, Y, Z")
# is inherently fragile because the model is a text generator, not an
# enum selector.  Even with temperature=0, the same prompt can produce:
#   "finance"        — correct
#   "Finance"        — case mismatch
#   "finance."       — trailing punctuation
#   "the domain is finance" — extra text
#   "I think it's finance" — conversational filler
#
# Each of these requires a brittle parsing heuristic (.strip().lower(),
# .split()[-1], regex), and each heuristic introduces a new failure mode.
#
# OpenAI function calling (tools parameter) gives us a machine-readable
# structured output contract.  The model MUST return a valid JSON object
# matching the schema, or the API itself rejects the response.  This:
#   1. Eliminates parsing entirely — we read result["domain"] directly.
#   2. Guarantees one of the three enum values (the API enforces enums).
#   3. Fails fast with a clean API error if the model cannot comply,
#      rather than silently returning garbage text.
#
# This is the standard pattern for classification in production LLM
# systems across the industry (OpenAI Cookbook, LangChain, LlamaIndex).

from typing import List, Dict, Tuple
import uuid
from openai import AsyncOpenAI
from .vectorstore import DomainVectorStore
from .logger import setup_logger

logger = setup_logger("documind.rag")

# The known domain keys.  Must match the keys in DomainVectorStore.docs_by_domain.
VALID_DOMAINS = {"finance", "healthcare", "legal"}

# Function definition for constrained domain classification.
# The OpenAI API enforces the enum constraint server-side, guaranteeing
# a parseable response regardless of model behaviour.
CLASSIFY_DOMAIN_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_domain",
        "description": "Classify a user's query into one of the supported knowledge domains.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "enum": ["finance", "healthcare", "legal"],
                    "description": "The domain that best matches the user's query.",
                }
            },
            "required": ["domain"],
            "additionalProperties": False,
        },
    },
}


class DocuMindRAG:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.vector_store = DomainVectorStore()
        self.max_retries = 2

    async def _classify_domain(self, query: str) -> str:
        """
        Classify a query into one of the supported domains using function calling.

        Returns one of: "finance", "healthcare", "legal", or "unknown" if the
        model cannot determine a domain (defense-in-depth fallback).
        """
        resp = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a domain classifier. Your ONLY job is to call the "
                        "classify_domain function with the appropriate domain. "
                        "Do not respond with any text — only call the function."
                    ),
                },
                {"role": "user", "content": query},
            ],
            # Function calling with tool_choice="required" forces the model to
            # call the function on every turn (rather than optionally generating
            # a text response).  This is the strictest constraint available.
            tools=[CLASSIFY_DOMAIN_TOOL],
            tool_choice={"type": "function", "function": {"name": "classify_domain"}},
            max_tokens=50,
            temperature=0.0,
        )

        msg = resp.choices[0].message

        # Defensive: verify the model actually made a tool call.
        # Under normal operation with tool_choice="required" this always
        # succeeds, but we guard against edge cases (API changes, model
        # regressions) with a fallback to "unknown".
        if msg.tool_calls:
            try:
                import json

                args = json.loads(msg.tool_calls[0].function.arguments)
                domain = args.get("domain", "unknown").lower()
                return domain if domain in VALID_DOMAINS else "unknown"
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                logger.warning(f"Failed to parse domain classification: {exc}")
                return "unknown"

        logger.warning(
            "Model returned no tool call during domain classification. "
            "This should not happen with tool_choice='required'. "
            f"Raw response: {msg.content}"
        )
        return "unknown"

    async def _evaluate_retrieval(
        self, query: str, docs: List[Tuple[Dict, float]]
    ) -> Tuple[bool, str]:
        doc_text = "\n".join(
            [f"ID: {d[0]['id']}\nContent: {d[0]['text']}" for d in docs]
        )
        prompt = (
            f"Does this context fully answer the query? Answer 'yes' or 'no'.\n"
            f"If 'no', provide: 'no\nReformulated query: ...'.\n\n"
            f"Query: {query}\nContext:\n{doc_text}"
        )
        resp = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        result = resp.choices[0].message.content.strip().lower()
        logger.info(f"Evaluator raw response: {result}")

        if result.startswith("yes"):
            return True, ""

        # Robust parsing: try to extract reformulated query, fall back to empty string
        reformulated = ""
        if "reformulated query:" in result:
            reformulated = result.split("reformulated query:")[-1].strip()
        return False, reformulated

    async def _generate_answer(
        self, query: str, domain: str, docs: List[Tuple[Dict, float]]
    ) -> str:
        context = "\n".join([f"Doc {d[0]['id']}: {d[0]['text']}" for d in docs])
        prompt = (
            f"You are a {domain} assistant. Use ONLY the context below to answer the query.\n"
            f"If the context is insufficient, say so without speculation.\n"
            f"Maximum 3 sentences.\n\nContext:\n{context}\n\nQuery: {query}"
        )
        resp = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    async def process_query(self, query: str) -> Dict:
        thought_process = []

        # 1. Domain routing
        domain = await self._classify_domain(query)
        thought_process.append(f"Domain classified as: {domain}")

        # 2. Initial retrieval
        docs = self.vector_store.retrieve(query, domain)
        if not docs:
            return {
                "answer": "No relevant documents found in the specified domain.",
                "domain": domain,
                "thought_process": thought_process,
                "retrieved_docs": [],
                "citations": [],
            }

        # 3. Self‑correction loop
        attempt = 0
        current_query = query
        while attempt <= self.max_retries:
            attempt += 1
            sufficient, reformulated = await self._evaluate_retrieval(
                current_query, docs
            )
            thought_process.append(
                f"Attempt {attempt}: Evaluator says {'sufficient' if sufficient else 'insufficient'}"
            )
            if sufficient:
                break
            if reformulated and attempt <= self.max_retries:
                thought_process.append(f"Reformulating query to: {reformulated}")
                new_docs = self.vector_store.retrieve(reformulated, domain)
                if new_docs:
                    docs = new_docs
                current_query = reformulated
            else:
                break

        # 4. Final answer
        answer = await self._generate_answer(query, domain, docs)

        # Build retrieved docs list for transparency
        retrieved_docs_list = [
            {
                "doc_id": d[0]["id"],
                "content": d[0]["text"],
                "relevance_score": round(float(d[1]), 4),
            }
            for d in docs
        ]

        return {
            "answer": answer,
            "domain": domain,
            "thought_process": thought_process,
            "retrieved_docs": retrieved_docs_list,
            "citations": list(set(d[0]["id"] for d in docs)),
        }
