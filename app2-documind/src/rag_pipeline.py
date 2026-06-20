# rag_pipeline.py

from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from .vectorstore import DomainVectorStore
from .logger import setup_logger

logger = setup_logger("documind.rag")


class DocuMindRAG:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.vector_store = DomainVectorStore()
        self.max_retries = 2

    async def _classify_domain(self, query: str) -> str:
        prompt = f"Classify the query into one domain: finance, healthcare, legal. Query: {query}\nDomain:"
        resp = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip().lower()

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
